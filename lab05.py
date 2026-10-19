

from pathlib import Path
import csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED_COUNT = 3
RESULTS, FIGURES = Path("results"), Path("figures")
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def save_csv(name, rows):
    if rows:
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)



import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4
NOISE_DIM = 2
N_INPUTS = N_CLASSES + NOISE_DIM


def make_recall_batch(batch_size, seq_len, generator):
    labels = torch.randint(0, N_CLASSES, (batch_size,), generator=generator)
    x = torch.randn(batch_size, seq_len, N_INPUTS, generator=generator) * 0.5
    x[:, :, :N_CLASSES] = 0.0
    x[torch.arange(batch_size), 0, labels] = 1.0
    return x, labels


def make_order_batch(batch_size, seq_len, generator, vocab=10):
    tokens = torch.randint(3, vocab, (batch_size, seq_len), generator=generator)
    pos = torch.stack([torch.randperm(seq_len, generator=generator)[:2] for _ in range(batch_size)])
    rows = torch.arange(batch_size)
    tokens[rows, pos[:, 0]] = 1
    tokens[rows, pos[:, 1]] = 2
    return tokens, (pos[:, 0] < pos[:, 1]).long()


def reference_attention(q, k, v, mask=None):

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), None


class ReferenceMHA(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        blocked = None if mask is None else ~mask
        return self.mha(x, context, context, attn_mask=blocked, need_weights=False)[0]


class AttentionReadout(nn.Module):

    def __init__(self, attention_fn=reference_attention, d=32):
        super().__init__()
        self.attention_fn = attention_fn
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.key, self.value = nn.Linear(N_INPUTS, d), nn.Linear(N_INPUTS, d)
        self.head = nn.Linear(d, N_CLASSES)

    def forward(self, x):
        q = self.query.expand(x.shape[0], -1, -1)
        out, weights = self.attention_fn(q, self.key(x), self.value(x))
        self.last_weights = weights                      # (batch, 1, T) when your function returns weights
        return self.head(out[:, 0])


class RecallLSTM(nn.Module):


    def __init__(self, hidden=32, forget_bias=3.0):
        super().__init__()
        self.rnn = nn.LSTM(N_INPUTS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_CLASSES)
        if forget_bias is not None:
            with torch.no_grad():
                self.rnn.bias_ih_l0[hidden:2 * hidden].fill_(forget_bias / 2)
                self.rnn.bias_hh_l0[hidden:2 * hidden].fill_(forget_bias / 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def train_recall(kind, seq_len, seed=0, steps=600, batch_size=64, lr=3e-3, clip=1.0, attention_fn=reference_attention):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    model = AttentionReadout(attention_fn) if kind == "attention" else RecallLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_recall_batch(batch_size, seq_len, generator)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    x_val, y_val = make_recall_batch(1000, seq_len, torch.Generator().manual_seed(999))
    with torch.no_grad():
        accuracy = (model(x_val).argmax(1) == y_val).float().mean().item()
        weights = getattr(model, "last_weights", None)
    return accuracy, (None if weights is None else weights[:, 0, 0].mean().item())


def sinusoidal_encoding(max_len, d_model):
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(position * freq), torch.cos(position * freq)
    return pe


class EncoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, norm, attention_cls):
        super().__init__()
        self.norm = norm
        self.attn = attention_cls(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        if self.norm == "pre":
            x = x + self.attn(self.ln1(x))
            return x + self.ff(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ff(x))


class OrderClassifier(nn.Module):

    def __init__(self, pos="sinusoidal", depth=2, norm="pre", attention_cls=ReferenceMHA,
                 d_model=32, n_heads=4, d_ff=64, vocab=10, max_len=64):
        super().__init__()
        self.pos_kind = pos
        self.embed = nn.Embedding(vocab, d_model)
        if pos == "learned":
            self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        elif pos == "sinusoidal":
            self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, norm, attention_cls) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d_model) if norm == "pre" else nn.Identity()
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens):
        x = self.embed(tokens)
        length = tokens.shape[1]
        if self.pos_kind == "learned":
            x = x + self.pos[:length]
        elif self.pos_kind == "sinusoidal":
            x = x + self.pe[:length]
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x).mean(dim=1))


def train_order(pos, seed, depth=2, norm="pre", warmup=0, steps=800, lr=2e-3, batch_size=64,
                train_len=16, test_lens=(16, 32), mixed_lens=None, clip=1.0, attention_cls=ReferenceMHA):

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    model = OrderClassifier(pos, depth, norm, attention_cls)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / warmup) if warmup else lr
        if mixed_lens is None:
            length = train_len
        else:
            length = mixed_lens[torch.randint(len(mixed_lens), (1,), generator=generator).item()]
        x, y = make_order_batch(batch_size, length, generator)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    accuracies = []
    with torch.no_grad():
        for length in test_lens:
            xv, yv = make_order_batch(1000, length, torch.Generator().manual_seed(999))
            accuracies.append((model(xv).argmax(1) == yv).float().mean().item())
    return accuracies


if __name__ == "__main__":
    accuracy, _ = train_recall("attention", seq_len=30, seed=0)
    print(f"Attention readout, 30 steps, 600 updates: validation accuracy {accuracy:.3f} (chance is 0.25)")
    assert accuracy > 0.9, "The starter should solve the 30-step recall task; check your environment."
    accuracies = train_order("sinusoidal", seed=0, steps=300, test_lens=(16,))
    print(f"Order task, 2 blocks, 300 updates: validation accuracy {accuracies[0]:.3f} (chance is 0.50)")
    assert accuracies[0] > 0.6, "The order task should be learnable; check your environment."
    print("Starter OK.")


# Experiment 1: custom attention and verification

def scaled_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__(); assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        b, tq, d = x.shape; tk = context.shape[1]
        q = self.qkv[0](x).view(b,tq,self.h,self.d).transpose(1,2)
        k = self.qkv[1](context).view(b,tk,self.h,self.d).transpose(1,2)
        v = self.qkv[2](context).view(b,tk,self.h,self.d).transpose(1,2)
        if mask is not None: mask = mask[None,None,:,:]
        y, _ = scaled_attention(q,k,v,mask)
        return self.out(y.transpose(1,2).contiguous().view(b,tq,d))

torch.manual_seed(1); q,k,v = [torch.randn(2,3,5,8) for _ in range(3)]
for mask in [None, torch.tril(torch.ones(5,5,dtype=torch.bool))]:
    a,w = scaled_attention(q,k,v,mask)
    ref = F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    assert torch.allclose(a,ref,atol=1e-5)
    assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-6)
    if mask is not None: assert torch.all(w.masked_select(~mask[None,None]) == 0)

mha = MultiHeadAttention(32,4); x = torch.randn(2,4,32); context = torch.randn(2,5,32)
assert mha(x).shape == x.shape
assert mha(x,mask=torch.tril(torch.ones(4,4,dtype=torch.bool))).shape == x.shape
assert mha(x,context).shape == (2,4,32)
# Parameter counts for head counts 1,2,4,8; permutation equivariance.
param_rows=[]
for h in (1,2,4,8):
    m=MultiHeadAttention(32,h); param_rows.append({"heads":h,"parameters":sum(p.numel() for p in m.parameters())})
perm=torch.randperm(4); assert torch.allclose(mha(x[:,perm]),mha(x)[:,perm],atol=1e-5)
save_csv("attention_parameters.csv",param_rows)
print("Attention tests passed. Parameter counts:",param_rows)



recall=[]
for length in (30,100,400):
    for seed in range(SEED_COUNT):
        for model in ("attention","lstm"):
            acc, w0 = train_recall(model,length,seed=seed,steps=600,attention_fn=scaled_attention if model=="attention" else None)
            recall.append({"length":length,"seed":seed,"model":model,"accuracy":acc,"attention_weight_step0":w0})
            print(recall[-1])
save_csv("recall_results.csv",recall)



position_rows=[]
for pos in ("none","learned","sinusoidal"):
    for seed in range(SEED_COUNT):
        acc=train_order(pos,seed=seed,steps=800,train_len=16,test_lens=(16,32),attention_cls=MultiHeadAttention)
        position_rows += [{"setting":"fixed","position":pos,"seed":seed,"test_length":L,"accuracy":A} for L,A in zip((16,32),acc)]
    for seed in range(SEED_COUNT):
        acc=train_order(pos,seed=seed,steps=800,mixed_lens=(8,16,24),test_lens=(16,32,48),attention_cls=MultiHeadAttention)
        position_rows += [{"setting":"mixed","position":pos,"seed":seed,"test_length":L,"accuracy":A} for L,A in zip((16,32,48),acc)]
save_csv("position_results.csv",position_rows)
for row in position_rows: print(row)



deep_rows=[]
for norm,warmup in (("post",0),("post",100),("pre",0)):
    for seed in range(SEED_COUNT):
        acc=train_order("sinusoidal",seed=seed,depth=12,norm=norm,warmup=warmup,steps=400,lr=3e-3,test_lens=(16,),attention_cls=MultiHeadAttention)
        deep_rows.append({"norm":norm,"warmup":warmup,"seed":seed,"accuracy_len16":acc[0]})
save_csv("deep_stack_results.csv",deep_rows)
print(deep_rows)

def best_time(model, x):
    times=[]
    for _ in range(3):
        model.zero_grad(set_to_none=True); start=time.perf_counter()
        model(x).sum().backward(); times.append(time.perf_counter()-start)
    return min(times)

timing=[]
for length in (100,200,400,800):
    x=torch.randn(32,length,32)
    models={"LSTM":nn.LSTM(32,32,batch_first=True),"Attention":MultiHeadAttention(32,4)}
    # LSTM outputs a tuple, so use a small wrapper.
    class LSTMOnly(nn.Module):
        def __init__(self,m): super().__init__(); self.m=m
        def forward(self,x): return self.m(x)[0]
    models["LSTM"]=LSTMOnly(models["LSTM"])
    for name,m in models.items():
        seconds=best_time(m,x); timing.append({"model":name,"length":length,"batch_size":32,"best_seconds":seconds})
save_csv("timing_results.csv",timing)
print(timing)


print("\nLab finished. Check the results/ and figures/ folders.")
