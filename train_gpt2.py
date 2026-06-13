import math
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        # 确保嵌入维度可以被注意力头整除
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 做一个mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # bs, seq_len, embd
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0 , float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) X (B, nh, T, hs) -> (B,  nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # softmax前的linear层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权值共享，词嵌入层的权重与语言模型头的权重一致
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.size()  # [batch_size, seq_len]
        assert T <= self.config.block_size, f"不能让seq_len {T} 大于 block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:  # 如果提供了目标值，那么计算损失
            # 将logits重塑成(B*T, vocab_size)
            # targets重塑成(B*T)
            # cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("从预训练的GPT中加载模型:", model_type)

        # 根据模型类型确认参数
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M param
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M param
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M param
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M param
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        # 创建GPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # 从huggingface/transformers模型中初始化
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 将参数逐一对齐并复制
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配, {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # openai使用了一个叫conv1d的模型，功能与linear一致，我们使用linear，需要单独处理它。需要转置
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:  # 其余的直接复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    # 参数正则化
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn ,p in self.named_parameters()}  # 获取所有参数
        param_dict = {pn: p for pn ,p in param_dict.items() if p.requires_grad}  # 过滤出需要梯度的参数
        # 全连接，embedding层的参数需要权重衰减，bias和layernorm不需要
        decay_param = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_param = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_param, "weight_decay": weight_decay},
            {"params": no_decay_param, "weight_decay": 0.0}
        ]
        # 打印一下需要做权重衰减的参数量和不需要做权重衰减的参数量
        num_decay_param = sum(p.numel() for p in decay_param)
        num_no_decay_param = sum(p.numel() for p in no_decay_param)
        if master_process:
            print(f"需要做权重衰减的参数量: {num_decay_param}")
            print(f"不需要做权重衰减的参数量: {num_no_decay_param}")

        # 检查adamW是否支持fused操作 
        # inspect.signature用来获取函数的签名
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"使用{'fused' if use_fused else 'python'}版本的AdamW优化器")
        # 初始化优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=[0.9, 0.95], eps=1e-8, fused=use_fused)
        return optimizer

import tiktoken
from transformers import BertTokenizer, AutoTokenizer
# 小型的数据加载器，并且可以生成批次
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        max_len = 1024
        tokens = []

        # enc = tiktoken.get_encoding("gpt2")
        # with open("shakespeare.txt", "r") as f:
        #     text = f.read()
        # tokens = enc.encode(text)
        # 加载斗破苍穹并tokenizer
        tokenizer = BertTokenizer.from_pretrained("model/tiansz/bert-base-chinese")
        with open("doup.txt", "r") as f:
            text = f.read()
        for i in range(0, len(text), max_len):
            chunk = text[i: i + max_len]
            tokenized_chunk = tokenizer([chunk])["input_ids"][0]
            tokens.extend(tokenized_chunk)
        self.tokens = torch.tensor(tokens)

        if master_process:
            # 打印数据加载信息
            print(f"加载了 {len(self.tokens)} tokens")
            print(f"epoch: {len(self.tokens) // (B*T)} batches")
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        # 获取下一个批次数据
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_process

        if self.current_position + (B * T * self.num_process + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# 初始化分布式数据并行DDP
# DDP 启动时不能使用python xx.py
# 需要使用
# torchrun --standalone --nproc_per_node=2 xx.py
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
ddp = int(os.environ.get("RANK", -1)) != -1 # 判断是否使用DDP
if ddp:
    assert torch.cuda.is_available(), "使用DDP必须使用GPU"
    init_process_group(backend="nccl") # 初始化进程组，使用nccl通信后端
    ddp_rank = int(os.environ["RANK"]) # 获取当前进程的rank
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # 获取进程总数
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # 获取当前进程本地rank
    device = f"cuda:{ddp_local_rank}" # 设置当前进程的设备,例如cuda:0   cuda:1
    torch.cuda.set_device(device) # 设置当前进程的GPU
    master_process = ddp_rank == 0 # 判断当前进程是否是主进程
else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
if master_process:
    print(f"使用的设备：{device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

import time

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288  # GPT3里的0.5M
B = 32
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, "确保total_batch_size是B*T*ddp_world_size的整数倍"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size) # 梯度累积步数
if master_process:
    print(f"梯度累积步数: {grad_accum_steps}")  
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_process=ddp_world_size)

# 设置张量精度为TF32
torch.set_float32_matmul_precision("high")

# model = GPT(GPTConfig(vocab_size=50304))  #  词表大小是50257
model = GPT(GPTConfig(vocab_size=21504))  #  词表大小是21127
# print("没报错！！！")
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # 如果使用DDP，那么model是DDP对象，module是原始模型

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 5000
# 学习率调度器
def get_lr(it):
    # 1、线性预热阶段
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2、超过最大迭代次数，使用最小学习率
    if it >= max_steps:
        return min_lr
    # 3、余弦退火阶段
    # decay_ratio从 1/(max_steps - warmup_steps) 到 1
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    # 计算余弦衰减系数
    # decay_ratio=0,coeff=1
    # decay_ratio=1,coeff=0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# 初始化优化器
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# 创建日志和模型的保存路径
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass


# ========== 断点续训：自动加载最新的完整 checkpoint ==========
resume_step = 0
if master_process:
    print("正在扫描可用的 checkpoint...")

# 获取 log_dir 下所有 model_*.pt 文件
import glob
checkpoint_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
# 提取 step 数字并排序（降序）
step_nums = []
for f in checkpoint_files:
    try:
        num = int(os.path.basename(f).split("_")[1].split(".")[0])
        step_nums.append((num, f))
    except:
        continue
step_nums.sort(reverse=True)  # 从大到小

# 依次尝试加载，直到成功
for step_num, ckpt_path in step_nums:
    try:
        if master_process:
            print(f"尝试加载 {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # 验证必要字段
        if "model" not in checkpoint or "step" not in checkpoint:
            continue
        raw_model.load_state_dict(checkpoint["model"])
        resume_step = checkpoint["step"] + 1

        # 恢复优化器（如果存在）
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if master_process:
                print(f"成功从 step {checkpoint['step']} 恢复（含优化器状态）")
        else:
            if master_process:
                print(f"成功从 step {checkpoint['step']} 恢复，将继续从 step {resume_step} 训练")

        # 可选：恢复 loss（如果需要记录）
        # last_loss = checkpoint.get("loss", None)       
        break
    except Exception as e:
        if master_process:
            print(f"加载 {ckpt_path} 失败: {e}，尝试下一个...")
        continue
else:
    if master_process:
        print("未找到可用的 checkpoint，从头开始训练")
# ============================================================


# 训练模型
for step in range(resume_step, max_steps):
    model.train()
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    last_step = (step == max_steps - 1)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # 如果不除以grad_accum_steps，那么累积的梯度就是实际梯度的grad_accum_steps倍
        loss_accum += loss.detach()
        if ddp:  # 进行梯度的同步
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # 梯度平均
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
    # 获取学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize() # 强制CPU等待GPU完成所有已提交的任务， 这句话会阻塞CPU，直到GPU完成所有操作
    t1 = time.time()
    dt = t1 - t0  # 计算每一步的时间,us
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
    if master_process:
        print(f"step {step:4d} |loss:{loss_accum.item():.6f}  |lr:{lr:.4e} |norm:{norm:.4f} |dt:{dt * 1000:.2f}ms |tokens/s:{tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"step {step:4d} |loss:{loss_accum.item():.6f}  |lr:{lr:.4e} |norm:{norm:.4f} |dt:{dt * 1000:.2f}ms |tokens/s:{tokens_per_sec:.2f}\n")
        if step > 0 and (step % 100 ==0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                "model":raw_model.state_dict(),
                "config":raw_model.config,
                "step":step,
                "loss":loss_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)

if ddp:
    destroy_process_group()  # 销毁进程组

import sys;sys.exit(0)

import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
print(tokens)
tokens = tokens.unsqueeze(0)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
max_length = 30
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

tokens = x[0, :max_length].tolist()
decoded = enc.decode(tokens)
print(">", decoded)


