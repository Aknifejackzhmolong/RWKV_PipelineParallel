from typing import Any

import torch
import torch.distributed as dist
from src.model import RWKV_RNN


class P2pLayerBegin(torch.autograd.Function):
    next_id = None
    prev_id = None
    n_embd = None
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if P2pLayerBegin.prev_id is not None:
            dist.recv(x,src=P2pLayerBegin.prev_id)
        return x
    @staticmethod
    def backward(ctx, grad_outputs):
        if P2pLayerBegin.prev_id is not None:
            dist.send(grad_outputs,dst=P2pLayerBegin.prev_id)
        # if P2pLayerBegin.next_id is None:
        #     print(f'rank[-1] backward state grad dispatch')
        return grad_outputs
class P2pLayerOffload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_func, x, state):
        ctx.run_func = run_func
        ctx.save_for_backward(x.cpu(),state.cpu())
        with torch.no_grad():
            x_out,state_out = ctx.run_func(x.clone(),state.clone())
        return x_out,state_out
    @staticmethod
    def backward(ctx, x_grad, state_grad):
        x,state = ctx.saved_tensors
        x,state = x.cuda(),state.cuda()
        with torch.enable_grad():
            x_out,state_out = ctx.run_func(x,state.clone())
            torch.autograd.backward((x_out,state_out),(x_grad, state_grad))
        return (None,x.grad,state.grad)
class P2pLayerEnd(torch.autograd.Function):
    next_id = None
    prev_id = None
    n_embd = None
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if P2pLayerEnd.next_id is not None:
            dist.send(x,dst=P2pLayerEnd.next_id)
        return x
    @staticmethod
    def backward(ctx, grad_outputs):
        if P2pLayerEnd.next_id is not None:
            grad_outputs = grad_outputs.contiguous()
            dist.recv(grad_outputs,src=P2pLayerEnd.next_id)
        #     print(f'rank[{P2pLayerEnd.next_id - 1}] backward')
        # else:
        #     print(f'rank[-1] backward start')
        return grad_outputs

class PipeSchedule:
    def __init__(self,model:RWKV_RNN):
        self.model = model
        self.rank_id = model.args.rank_id
        self.prev_id = model.args.prev_id
        self.next_id = model.args.next_id
        self.world_size = model.args.world_size
        P2pLayerBegin.prev_id = P2pLayerEnd.prev_id = model.args.prev_id
        P2pLayerBegin.next_id = P2pLayerEnd.next_id = model.args.next_id
        P2pLayerBegin.n_embd = P2pLayerEnd.n_embd = model.n_embd

        self.output_tensors = []
        self.stack_tensors = []
    def forward(self,x,state):
        if self.prev_id is not None:
            num_token = len(x[0])
            x = torch.zeros((1,num_token,P2pLayerBegin.n_embd),dtype=self.model.datatype,requires_grad=True).cuda()
        # print(f'RANK[{self.rank_id}] forward begin')
        x = P2pLayerBegin.apply(x)
        x,state = self.model.forward_parallel(x, state)
        # x,state = P2pLayerOffload.apply(self.model.forward_parallel, x, state)
        x = P2pLayerEnd.apply(x)
        # print(f'RANK[{self.rank_id}] forward end')
        self.output_tensors.append(x.sum())
        return x, state#.detach_()
    def backward(self):
        # TODO: 为RWKV或LSTM这类有state保存上个epoch计算图的方法打造
        x = self.output_tensors[0]
        self.output_tensors = self.output_tensors[1:]
        self.stack_tensors = [x] + self.stack_tensors
        print(f'RANK[{self.rank_id}] backward begin {len(self.stack_tensors)}')
        for x in self.stack_tensors:
            x.backward(retain_graph=True)
        print(f'RANK[{self.rank_id}] backward end')
    def backward_without_state(self,*args: Any, **kwargs: Any):
        # TODO: 废弃的方法，只能用在没有state存储之前epoch计算图的情况
        x = self.output_tensors[0]
        self.output_tensors = self.output_tensors[1:]
        # print(f'RANK[{self.rank_id}] backward begin')
        x.backward(*args,**kwargs)
        # print(f'RANK[{self.rank_id}] backward end')
    def train_with_gpipe(self,x,y,loss_fn):
        # todo: 未完成
        batch_size = len(x)
        self.output_tensors = []
        self.stack_tensors = []
        state = torch.zeros((batch_size,self.model.block_num * (self.model.head_size+2),self.model.n_embd))
        start = 0
        if self.next_id is None:
            for i in range(batch_size):
                x_out,state = self.forward(x,state)
                loss = loss_fn(x_out,y[i])
                loss.backward(retain_graph=True)
                del self.output_tensors[0]
        else:
            while start < batch_size:
                parallel_size = min(batch_size - start,self.world_size)
                for _ in range(parallel_size):
                    self.forward(x,state)
                for _ in range(parallel_size):
                    self.backward()
                start += parallel_size
    def train_with_interleaving(self,x,y,state,loss_fn):
        分段长度=self.model.args.token_limit
        num_tok = torch.tensor([len(x)])
        input_mask = torch.arange(torch.ceil((num_tok - 1) / 分段长度).item() + 1,dtype=torch.long) * 分段长度
        input_mask[-1] = min(input_mask[-1],num_tok)
        batch_size = len(input_mask) - 1
        self.output_tensors = []
        self.stack_tensors = []
        loss_total = torch.tensor([0.0],dtype=self.model.datatype).cuda()
        if self.next_id is None:
            for i in range(batch_size):
                start,end = tuple(input_mask[i:i+2])
                x_out,state = self.forward(x[None,start:end],state)
                loss = loss_fn(x_out[0],y[start:end])
                loss.backward(retain_graph=True)
                loss_total += loss.item()
        else:
            warm_up_size = min(batch_size,self.world_size) - 1 - self.rank_id
            warm_up_size = max(0,warm_up_size)
            # warmup step
            for i in range(warm_up_size):
                start,end = tuple(input_mask[i:i+2])
                self.forward(x[None,start:end],state)
            # 1f1b step
            for i in range(batch_size - warm_up_size):
                start,end = tuple(input_mask[warm_up_size+i:warm_up_size+i+2])
                self.forward(x[None,start:end],state)
                self.backward()
            # cooldown step
            for i in range(warm_up_size):
                self.backward()
        dist.broadcast(loss_total,self.world_size - 1)
        return loss_total
