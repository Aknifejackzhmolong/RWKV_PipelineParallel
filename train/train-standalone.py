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
from torch.utils.data import IterableDataset,DataLoader
from tqdm import tqdm
import json

class OffloadLayer(torch.autograd.Function):
    single = torch.tensor([0.],dtype=torch.float).cuda()
    @staticmethod
    def forward(ctx, run_func, x, state):
        ctx.run_func = run_func
        ctx.g = None
        ctx.save_for_backward(x.cpu(),state.cpu())
        with torch.no_grad():
            x_out, state_out = run_func(x, state)
        return x_out,state_out,OffloadLayer.single.clone()
    @staticmethod
    def backward(ctx, x_grad, state_grad, single):
        if single == 0: # loss.backward()默认1
            if ctx.g is None:
                ctx.g = state_grad.cpu()
            else:
                ctx.g = ctx.g + state_grad.cpu()
            return None,None,None
        if ctx.g is not None:
            state_grad = ctx.g.cuda() + state_grad
        x, state = ctx.saved_tensors
        x, state = x.cuda(), state.cuda()
        with torch.enable_grad():
            x_out, state_out = ctx.run_func(x, state.clone())
        torch.autograd.backward((x_out, state_out), (x_grad, state_grad))
        return None, x.grad, state.grad

class TextDataset(IterableDataset):
    def __init__(self, file_path,tokenizer):
        """
        Args:
            file_path:
            tokenizer:
        """
        self.file_path = file_path
        self.tokenizer = tokenizer

    def __iter__(self):
        file_path = self.file_path
        tokenizer = self.tokenizer
        token_limit = 128
        start = 512
        with open(file_path, "r", encoding='utf8') as file:
            while start < 12800:
                end = start + token_limit
                for line in file:
                    data = json.loads(line)
                    texts=data["text"]
                    lt = len(texts)
                    if lt <= start or lt > end:
                        continue
                    token = tokenizer.encode(texts)[0]+[0]
                    token = torch.tensor(token,dtype=torch.long)
                    yield token
                start = end

class TokenLimitModel(nn.Module):
    def __init__(self, model, loss_fn, token_limit):
        super(TokenLimitModel, self).__init__()
        self.model = model
        self.loss_fn_ = loss_fn
        self.token_limit = token_limit
    def forward(self, x, y, length_list, state):
        分段长度 = self.token_limit
        num_tok = length_list.sum()
        max_num_tok = length_list.max()
        input_mask = torch.arange(torch.ceil((max_num_tok - 1) / 分段长度).item() + 1, dtype=torch.long) * 分段长度
        input_mask[-1] = min(input_mask[-1], max_num_tok)
        loop_size = len(input_mask) - 1
        loss_stask = []
        loss_total = 0.0
        y = y.clone()
        for i,num in enumerate(length_list):
            y[i,num:] = -1
        for i in range(loop_size):
            start,end = tuple(input_mask[i:i+2])
            x_out,state,single = OffloadLayer.apply(self.model.forward_parallel,x[:,start:end],state)
            x_out = torch.concatenate(tuple(x_out), dim=0)
            y_out = torch.concatenate(tuple(y[:,start:end]), dim=0)
            x_out = x_out[y_out != -1]
            y_out = y_out[y_out != -1]
            loss = self.loss_fn_(x_out,y_out) + single
            loss_stask.append(loss)
            loss_total += loss.item() * len(y_out)
        for i,loss in enumerate(reversed(loss_stask)):
            loss.backward(retain_graph=True)
        return loss_total / num_tok

def collate_fn(token_list):
    length_list = [ len(x) - 1 for x in token_list ]
    maximum_size = max(length_list) + 1
    batch_data = torch.zeros((len(length_list),maximum_size), dtype=torch.long)
    for i,x in enumerate(token_list):
        batch_data[i,:len(x)] = x
    return batch_data, torch.tensor(length_list, dtype=torch.long)

def main(args:ModelArgs):
    device = torch.device(args.device)
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER(args.TOKENIZER_PATH)
    print("Done.")

    file_path = args.DATASET_PATH  # 替换为你的文本文件路径
    # 设置续写的初始字符串和参数
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    model.train()
    dataset = TextDataset(file_path,tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=1, prefetch_factor=args.batch_size*2)
    state = model.init_state(batch_size=args.batch_size).to(device)
    state.requires_grad = True
    model = TokenLimitModel(model,criterion,args.token_limit)
    with torch.autograd.set_detect_anomaly(True):
        with tqdm(dataloader, total=13e6 // args.batch_size) as tbar:
            for i, (token, length_list) in enumerate(tbar):
                token = token.cuda()
                x = token[:,:-1]
                y = token[:,1:]
                optimizer.zero_grad()
                loss = model(x,y,length_list,state.clone())
                tbar.set_postfix(loss=loss.item())
                optimizer.step()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                elif args.device == 'musa':
                    torch.musa.empty_cache()
                if i % 5000 == 0:
                    model.model.save_model(f'weight/checkpoint-small-75-{i}.pth')

    # 同步GPU执行位置
    end_time = time.time()
    model.model.save_model(f'weight/checkpoint-small-final.pth')
    save_time = time.time()

    # 计算并打印程序运行时间
    execution_time = end_time - start_time
    save_time = save_time - end_time
    dist.barrier()
    print(f"RANK[{args.rank_id}]程序运行时间：{execution_time:.2f}秒，保存模型耗时：{save_time:.2f}秒\n",end='')

if __name__ == '__main__':

    # 初始化模型参数
    with open("train/params-small.json", "r") as f:
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
    main(args)
