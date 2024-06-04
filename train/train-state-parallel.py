import sys
import os
import json
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')

# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))

import time
import torch
import torch.distributed as dist
from src.model import RWKV_RNN, ModelArgs
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
from train.PipeSchedule import PipeSchedule
class TextDataset(Dataset):
    def __init__(self,file_path:str,tokenizer):
        """
        Args:
            x (list[list[int]]): 预处理后的文本数据，每个样本是一个由单词索引组成的列表。
            y (list[list[int]]): 预处理后的文本数据，每个样本是一个由单词索引组成的列表。
        """
        data_all = []
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                texts=data["text"]
                token_list = (tokenizer.encode(texts)[0]+[0])
                data_all.append(token_list)
        self.data_all = data_all

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_all[idx],dtype=torch.long)
        x=data[:-1]
        y=data[1:]
        return x,y


def main(args:ModelArgs):
    device = torch.device(args.device)
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER(args.TOKENIZER_PATH)
    print("Done.")

    # 设置续写的初始字符串和参数
    if args.rank_id == 0:
        dataset = TextDataset(args.DATASET_PATH,tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        datasize = torch.tensor([len(dataloader)]).cuda()
    else:
        datasize = torch.tensor([0]).cuda()
        dataloader = None
    wrapper = PipeSchedule(model)
    # 根据rank为0的进程广播tensor
    dist.broadcast(datasize, 0)  # 其他进程接收广播的tensor
    datasize = datasize.item()
    if dataloader is None:
        x = y = torch.tensor([0])
        dataloader = [(x,y)] * datasize
    state_init, gather_list = init_state(model)
    state_init.requrire_grad = True
    optimizer = torch.optim.AdamW([state_init], lr=1.0)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.01, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    print(f"RANK[{args.rank_id}] state_size:{state_init.size()}") # 这里打印状态的形状
    torch.cuda.synchronize()    # 开始计时
    start_time = time.time()
    with torch.autograd.set_detect_anomaly(True):
        model.eval()
        warm_up_size = args.world_size - 1 - args.rank_id
        warm_up_size = max(0,warm_up_size)
        with tqdm(dataloader,disable=(args.rank_id < args.world_size - 1)) as tbar:
            last_list = [False]
            state = state_init.clone()
            loss_total = torch.tensor([0.0],dtype=wrapper.model.datatype).cuda()
            for step,x,y,last in generator(args,tbar):
                if wrapper.next_id is None:
                    x_out,state = wrapper.forward(x[None,:],state)
                    loss = criterion(x_out[0],y)
                    loss.backward(retain_graph=True)
                    loss_total[0] = loss.item()
                    del wrapper.output_tensors[0]
                    last_list = [last]
                else:
                    last_list += [last]
                    if step < warm_up_size:
                        # warmup step
                        wrapper.forward(x[None,:],state)
                    elif x is not None:
                        # 1f1b step
                        wrapper.forward(x[None,:],state)
                        wrapper.backward(retain_graph=True)
                        del last_list[0]
                    else:
                        # cooldown step
                        wrapper.backward(retain_graph=True)
                        del last_list[0]
                tbar.set_postfix(avg_loss=loss_total.item(), lr=optimizer.param_groups[0]['lr'])
                if last_list[0]:
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif args.device == 'musa':
                        torch.musa.empty_cache()
                if last:
                    state = state_init.clone()

    # 同步GPU执行位置
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算并打印程序运行时间
    execution_time = end_time - start_time
    if args.rank_id == 0:
        dist.gather(state_init,gather_list=gather_list,dst=0)
        state_init = torch.concatenate(gather_list,dim=1)
        model.save_state(state_init, "weight/state-trained-latest.pth")
    else:
        dist.gather(state_init,dst=0)
    print(f"RANK[{args.rank_id}]程序运行时间：{execution_time:.2f}秒\n",end='')

def boardcast_iter(x, y):
    if args.prev_id is None:
        num_tok = torch.tensor([len(x)]).cuda()
        dist.broadcast(num_tok,0)
    else:
        num_tok = torch.tensor([0]).cuda()
        dist.broadcast(num_tok,0)
        x = y = torch.zeros((num_tok,)).long().cuda()
    dist.broadcast(y, 0)
    return x,y


def generator(args, loader):
    warm_up_size = args.world_size - 1 - args.rank_id
    warm_up_size = max(0,warm_up_size)
    分段长度 = args.token_limit
    num_tok = torch.tensor([0]).cuda()
    step = 0
    for x, y in loader:
        x = x[0].cuda()
        y = y[0].cuda()
        if args.prev_id is None:
            num_tok[0] = len(x)
            dist.broadcast(num_tok, 0)
        else:
            dist.broadcast(num_tok, 0)
            x = y = torch.zeros((num_tok,)).long().cuda()
        dist.broadcast(y, 0)
        input_mask = torch.arange(torch.ceil((num_tok - 1) / 分段长度).item(), dtype=torch.long)
        input_mask = input_mask * 分段长度
        for i in range(len(input_mask) - 1):
            start = input_mask[i]
            end = input_mask[i + 1]
            yield step,x[start:end], y[start:end], False
            step += 1
        yield step,x[input_mask[-1]:], y[input_mask[-1]:], True
        step += 1
    for i in range(warm_up_size):
        yield step, None, None, False
        step += 1

def init_state(model:RWKV_RNN) -> torch.Tensor:
    slice_size = model.block_num * (2 + model.head_size)
    state = torch.empty((1, slice_size, model.n_embd),dtype=model.datatype).cuda()
    scatter_list = None
    if args.rank_id == 0:
        state_init = model.init_state(batch_size=1).cuda()
        scatter_list = []
        for i in range(args.world_size - 1):
            scatter_list += [state_init[:, (i * slice_size):((i + 1) * slice_size), :]]
        scatter_list += [state_init[:, ((args.world_size - 1) * slice_size):, :]]
        dist.scatter(state, scatter_list=scatter_list, src=0)
    else:
        dist.scatter(state, src=0)
    return state, scatter_list
def init_process(args:ModelArgs):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500' # 一机多卡请注释这行
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(torch.distributed.get_rank())
    args.rank_id = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    if args.rank_id == 0:
        args.prev_id = None
        args.next_id = args.rank_id + 1
    elif args.rank_id == args.world_size - 1:
        args.prev_id = args.rank_id - 1
        args.next_id = None
    else:
        args.prev_id = args.rank_id - 1
        args.next_id = args.rank_id + 1

if __name__ == '__main__':

    # 初始化模型参数
    with open("train/params.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))
        assert args.device in ['cpu', 'cuda', 'musa', 'npu']
        # 如果是国产硬件，需要 import 插件来 hack pytorch
        if args.device == "musa":
            import torch_musa
        elif args.device == "npu":
            import torch_npu
        # try musa/cuda :P
        try:
            if torch.cuda.is_available():
                args.device = 'cuda'
            else:
                import torch_musa
                if torch.musa.is_available():
                    args.device = 'musa'
        except:
            pass
    init_process(args)
    main(args)
