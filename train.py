"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


class Muon(torch.optim.Optimizer):
    """Small dense-matrix Muon optimizer using Newton-Schulz orthogonalization."""
    def __init__(self, params, lr=0.01, momentum=0.95, weight_decay=0.1, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     weight_decay=weight_decay, ns_steps=ns_steps))

    @staticmethod
    def _orthogonalize(update, steps):
        # Newton-Schulz works best with rows <= columns; restore orientation
        # afterward so the returned update matches the parameter's shape.
        rows, cols = update.shape
        transposed = rows > cols
        x = update.float().clone()
        if transposed:
            x = x.T.contiguous()
        x = x / x.norm().clamp_min(1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            gram = x @ x.T
            x = a * x + (b * gram + c * (gram @ gram)) @ x
        if transposed:
            x = x.T.contiguous()
        # Compensate for rectangular matrix aspect ratio after restoring shape.
        x = x * max(1.0, rows / cols) ** 0.5
        return x.to(dtype=update.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta = group['momentum']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients.')
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                buffer = state['momentum_buffer']
                buffer.lerp_(grad, 1.0 - beta)
                # Nesterov-style momentum, followed by the matrix update's
                # approximate orthogonalization.
                update = grad.lerp(buffer, beta)
                update = self._orthogonalize(update, group['ns_steps'])
                if group['weight_decay']:
                    parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                parameter.add_(update, alpha=-group['lr'])
        return loss


def main():
    total_started = time.perf_counter() # 记录开始时间

    ## 创建命令行参数解析器
    p = argparse.ArgumentParser(description=__doc__)
    # 添加参数名称，类型和默认
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    # 参数合法性检查
    if args.steps < 1 or args.batch_size < 1: # step 和 batch size 必须大于 0 避免无意义训练
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()): # run-dir 不能已存在非空目录
        p.error('Run directory already contains results. Use a new --run-dir.')

    ## 初始化设备和随机种子
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)

    prepared = time.perf_counter()
    # 读取数据
    data = load_data()
    # 模型的参数配置     
    config = json.loads(args.config.read_text()) # jason 词典转 python 词典
    # 创建模型
    model, implementation_sha = make_model(args.implementation, config, device)
    # 创建运行存储文件夹
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # try11 hybrid optimizer: use Muon for transformer matrix weights, where
    # orthogonalized matrix updates are intended; keep embeddings, norms, and
    # other vectors on AdamW. The tied output head is the same token embedding
    # parameter and therefore also stays on AdamW.
    if args.implementation.lower() == 'student':
        muon_parameters, adamw_parameters = [], []
        for name, parameter in model.named_parameters():
            if name.startswith('blocks.') and parameter.ndim == 2:
                muon_parameters.append(parameter)
            else:
                adamw_parameters.append(parameter)
        optimizers = []
        if muon_parameters:
            optimizers.append((Muon(muon_parameters, lr=.01, weight_decay=.1), .01))
        if adamw_parameters:
            optimizers.append((torch.optim.AdamW(adamw_parameters, lr=.001,
                                                 weight_decay=.1), .001))
    else:
        # Preserve the original training recipe for the other implementations.
        optimizers = [(torch.optim.AdamW(model.parameters(), lr=.001,
                                         weight_decay=.1), .001)]
    tokens = data['train'][0].to(device)
    # 创建随机数生成器,同一个 seed 可以保证同样的训练顺序
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    preparation_seconds = time.perf_counter()-prepared
    
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for optimizer, base_lr in optimizers:
            for group in optimizer.param_groups:
                group['lr'] = base_lr * learning_rate / .001
            optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            # try11 removes the MTP auxiliary objective: all implementations
            # are trained on the standard one-step causal language-model loss.
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for optimizer, _ in optimizers:
            optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
