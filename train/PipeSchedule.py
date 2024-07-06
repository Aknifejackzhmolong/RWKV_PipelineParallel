from typing import Any

import torch
import torch.distributed as dist
from src.model import RWKV_RNN


class P2pLayerBegin(torch.autograd.Function):
    next_id = None
    prev_id = None
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        if P2pLayerBegin.prev_id is not None:
            dist.recv(x,src=P2pLayerBegin.prev_id)
        single = torch.tensor([0.0],dtype=torch.float,requires_grad=True)
        return x,single
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor, single: torch.Tensor):
        # pd = P2pLayerBegin.prev_id
        # pd = 0 if pd is None else (pd + 1)
        # print(f'rank[{pd}] backward state grad dispatch {single}')
        if P2pLayerBegin.prev_id is not None and single > 0:
            dist.send(grad_outputs,dst=P2pLayerBegin.prev_id)
            # print(f'rank[{pd}] backward state grad dispatch end')
        return grad_outputs
class OffloadLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_func, x, state, single):
        ctx.run_func = run_func
        ctx.g = None
        ctx.save_for_backward(x.cpu(),state.cpu())
        with torch.no_grad():
            x_out, state_out = run_func(x, state)
        return x_out,state_out,single
    @staticmethod
    def backward(ctx, x_grad, state_grad, single):
        if single == 0:
            if ctx.g is None:
                ctx.g = state_grad.cpu()
            else:
                ctx.g = ctx.g + state_grad.cpu()
            return None,None,None,single
        if ctx.g is not None:
            state_grad = ctx.g.cuda() + state_grad
        x, state = ctx.saved_tensors
        x, state = x.cuda(), state.cuda()
        if P2pLayerBegin.prev_id is not None:
            x.requires_grad = True
        state.requires_grad = True
        with torch.enable_grad():
            x_out, state_out = ctx.run_func(x, state.clone())
        torch.autograd.backward((x_out, state_out), (x_grad, state_grad))
        return (None, x.grad, state.grad, single)
class P2pLayerEnd(torch.autograd.Function):
    next_id = None
    prev_id = None
    @staticmethod
    def forward(ctx, x, single):
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
        single = torch.tensor([1.0],dtype=torch.float)
        return grad_outputs,single

class PipeSchedule:
    def __init__(self,model:RWKV_RNN):
        self.model = model
        self.rank_id = model.args.rank_id
        self.prev_id = model.args.prev_id
        self.next_id = model.args.next_id
        self.world_size = model.args.world_size
        P2pLayerBegin.prev_id = P2pLayerEnd.prev_id = model.args.prev_id
        P2pLayerBegin.next_id = P2pLayerEnd.next_id = model.args.next_id
        self.output_tensors = []
    def forward(self,x,state):
        if self.prev_id is not None:
            num_token = len(x[0])
            x = torch.zeros((1,num_token,self.model.n_embd),dtype=self.model.datatype,requires_grad=True).cuda()
        if not state.requires_grad:
            state.requires_grad = True
        # print(f'RANK[{self.rank_id}] forward begin')
        x,single = P2pLayerBegin.apply(x)
        # x_out,state_out = self.model.forward_parallel(x, state)
        x_out,state_out,single = OffloadLayer.apply(self.model.forward_parallel, x, state, single)
        # x_out,state_out = torch.utils.checkpoint.checkpoint(self.model.forward_parallel, x, state)
        x_out = P2pLayerEnd.apply(x_out,single)
        # print(f'RANK[{self.rank_id}] forward end')
        self.output_tensors.append(x_out.sum())
        return x_out,state_out#.detach()
    def backward(self,*args: Any, **kwargs: Any):
        # TODO: 废弃的方法，只能用在没有state存储之前epoch计算图的情况
        x = self.output_tensors[0]
        # print(f'RANK[{self.rank_id}] backward begin')
        x.backward(*args,**kwargs)
        # print(f'RANK[{self.rank_id}] backward end')
        del self.output_tensors[0]
    def train_with_gpipe(self,x,y,state,loss_fn):
        分段长度=self.model.args.token_limit
        num_tok = torch.tensor([len(x)])
        input_mask = torch.arange(torch.ceil((num_tok - 1) / 分段长度).item() + 1,dtype=torch.long) * 分段长度
        input_mask[-1] = min(input_mask[-1],num_tok)
        batch_size = len(input_mask) - 1
        self.output_tensors = []
        pre_num = 0
        loss_total = torch.tensor([0.0],dtype=self.model.datatype).cuda()
        if self.next_id is None:
            output_tensors = []
            for i in range(batch_size):
                start,end = tuple(input_mask[i:i+2])
                cur_num = end - start
                x_out,state = self.forward(x[None,start:end],state)
                loss = loss_fn(x_out[0],y[start:end])
                output_tensors.append(loss)
                loss_total = loss * cur_num / (pre_num + cur_num) + loss_total.clone() * pre_num / (pre_num + cur_num)
                pre_num += cur_num
            self.output_tensors = list(reversed(output_tensors))
            for _ in range(batch_size):
                self.backward(retain_graph=True)
        else:
            for i in range(batch_size):
                start,end = tuple(input_mask[i:i+2])
                self.forward(x[None,start:end],state)
            self.output_tensors = list(reversed(self.output_tensors))
            for _ in range(batch_size):
                self.backward(retain_graph=True)
        dist.broadcast(loss_total.detach(),self.world_size - 1)
        return loss_total
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
        # dist.broadcast(loss_total,self.world_size - 1)
        return loss_total
