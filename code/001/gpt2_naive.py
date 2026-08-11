import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from tokenizers import Tokenizer


class Embedding(nn.Module):
    def __init__(self, token_weight, position_weight):
        super().__init__()
        self.token_embedding = nn.Embedding.from_pretrained(token_weight, freeze=True)
        self.position_embedding = nn.Embedding.from_pretrained(position_weight, freeze=True)

    # 方法名固定叫 forward：后面写 self.embed(input_ids) 时，PyTorch 执行的就是它
    def forward(self, input_ids):
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        # 这里只生成一份位置向量；相加时，PyTorch 会沿 batch 维自动广播。
        return token_embeds + position_embeds


class LayerNorm(nn.Module):
    def __init__(self, weight, bias, eps: float = 1e-5):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        # dim=-1 表示只在最后一维（768）上算，即每个 token 各算各的；keepdim 保留该维以便相减
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, W_qkv, b_qkv, W_proj, b_proj, n_head: int):
        super().__init__()
        self.W_qkv = W_qkv
        self.b_qkv = b_qkv
        self.W_proj = W_proj
        self.b_proj = b_proj
        self.n_head = n_head

    def forward(self, x):
        B, T, n_embd = x.shape
        head_dim = n_embd // self.n_head

        qkv = x @ self.W_qkv + self.b_qkv
        q, k, v = qkv.split(n_embd, dim=-1)

        # view 把 768 按 12×64 切成每个 head 一份，transpose 再把 head 维提到 T 前面，
        # 形状变成 [B, 12, T, 64]，后面的矩阵乘法就在每个 head 内部各算各的。
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        # 未来位置填成极小值，softmax 后权重会变成 0。
        scores = scores.masked_fill(
            ~mask.view(1, 1, T, T),
            torch.finfo(scores.dtype).min,
        )

        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ v
        # 把 12 个 head 的 64 维结果沿最后一维拼回 768 维。
        output = output.transpose(1, 2).contiguous().view(B, T, n_embd)
        return output @ self.W_proj + self.b_proj


class MLP(nn.Module):
    def __init__(self, W_fc, b_fc, W_proj, b_proj):
        super().__init__()
        self.W_fc = W_fc
        self.b_fc = b_fc
        self.W_proj = W_proj
        self.b_proj = b_proj

    @staticmethod
    def gelu_new(x):
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))
            )
        )

    def forward(self, x):
        x = x @ self.W_fc + self.b_fc
        x = self.gelu_new(x)
        return x @ self.W_proj + self.b_proj


class Block(nn.Module):
    def __init__(self, weights, layer_idx, n_head, eps=1e-5):
        super().__init__()
        self.ln_1 = LayerNorm(
            weight=weights[f"h.{layer_idx}.ln_1.weight"],
            bias=weights[f"h.{layer_idx}.ln_1.bias"],
            eps=eps)
        self.attn = CausalSelfAttention(
            W_qkv=weights[f"h.{layer_idx}.attn.c_attn.weight"],
            b_qkv=weights[f"h.{layer_idx}.attn.c_attn.bias"],
            W_proj=weights[f"h.{layer_idx}.attn.c_proj.weight"],
            b_proj=weights[f"h.{layer_idx}.attn.c_proj.bias"],
            n_head=n_head)
        self.ln_2 = LayerNorm(
            weight=weights[f"h.{layer_idx}.ln_2.weight"],
            bias=weights[f"h.{layer_idx}.ln_2.bias"],
            eps=eps)
        self.mlp = MLP(
            W_fc=weights[f"h.{layer_idx}.mlp.c_fc.weight"],
            b_fc=weights[f"h.{layer_idx}.mlp.c_fc.bias"],
            W_proj=weights[f"h.{layer_idx}.mlp.c_proj.weight"],
            b_proj=weights[f"h.{layer_idx}.mlp.c_proj.bias"])

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, weights, n_layer, n_head, eps=1e-5):
        super().__init__()
        self.embed = Embedding(weights["wte.weight"], weights["wpe.weight"])
        self.lm_head_weight = weights["wte.weight"]
        self.blocks = nn.ModuleList(
            [Block(weights, layer_idx=i, n_head=n_head, eps=eps)
             for i in range(n_layer)]
        )
        self.ln_f = LayerNorm(
            weight=weights["ln_f.weight"],
            bias=weights["ln_f.bias"],
            eps=eps)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x @ self.lm_head_weight.T

    @torch.inference_mode()
    def inference(self, input_ids):
        logits = self(input_ids)
        last_logits = logits[:, -1, :]
        return torch.argmax(last_logits, dim=-1, keepdim=True)


def main():
    model_dir = Path(__file__).resolve().parent / "models/gpt2"
    prompt = "The meaning of life is"

    config = json.loads((model_dir / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    weights = load_file(str(model_dir / "model.safetensors"), device="cpu")

    model = GPT2(
        weights=weights,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        eps=config["layer_norm_epsilon"],
    )
    model.eval()

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long)

    max_new_tokens = 8
    output_ids = input_ids

    for _ in range(max_new_tokens):
        next_id = model.inference(output_ids)
        output_ids = torch.cat([output_ids, next_id], dim=1)

    output_text = tokenizer.decode(output_ids[0].tolist())

    print(output_ids.tolist())
    # [[464, 3616, 286, 1204, 318, 407, 262, 976, 355, 262, 3616, 286, 1918]]
    print(output_text)
    # The meaning of life is not the same as the meaning of death


if __name__ == "__main__":
    main()
