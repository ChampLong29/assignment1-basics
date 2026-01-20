import math
import torch
from jaxtyping import Float, Int

class Linear(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)

class Embedding(torch.nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ) -> None:

        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None=None
        ) -> None:

        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ) -> None:

        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.einsum('...d,fd->...f', x, self.w1)
        activation = torch.einsum('...d,fd->...f', x, self.w3)

        hidden = gate * torch.sigmoid(gate) * activation
        return torch.einsum('...f,df->...d', hidden, self.w2)

class RoPE(torch.nn.Module):
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k // 2, device=device, dtype=dtype)
        freqs = 1.0 / (theta ** (2 * k / d_k))

        positions = torch.arange(0, max_seq_len, device=device, dtype=dtype)

        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer(persistent=False, name="cos_cached", tensor=cos)
        self.register_buffer(persistent=False, name="sin_cached", tensor=sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 从缓存中提取对应位置的 cos 和 sin
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k//2)
        
        # 分离偶数和奇数维度（配对）
        x_even = x[..., 0::2]  # (..., seq_len, d_k//2) - 索引 0,2,4,...
        x_odd = x[..., 1::2]   # (..., seq_len, d_k//2) - 索引 1,3,5,...
        
        # 应用旋转变换
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos
        
        # 交错合并回原始形状
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # (..., seq_len, d_k)
        
        return x_rotated

def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    max_vals = torch.max(x, dim = dim, keepdim=True).values
    exp_vals = torch.exp(x - max_vals)
    return exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)

def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)

def cross_entropy(
    inputs: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, "batch_size"]
) -> Float[torch.Tensor, ""]:
    """
    Numerically stable cross-entropy loss.
    Uses log-sum-exp trick to avoid overflow.
    """
    # 数值稳定的 log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    max_vals = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    
    # 获取目标类的 logits
    batch_size = inputs.shape[0]
    target_logits = inputs[torch.arange(batch_size, device=inputs.device), targets]
    
    # cross_entropy = -log(softmax(logits)[target]) = -target_logit + max + log_sum_exp
    losses = -target_logits + max_vals.squeeze(-1) + log_sum_exp
    
    return losses.mean()

def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients to have maximum L2 norm of max_l2_norm.
    Modifies gradients in-place.
    """
    # 收集所有需要梯度的参数的梯度
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)
    
    if not grads:
        return
    
    # 计算总的 L2 范数
    total_norm_sq = torch.stack([torch.sum(g ** 2) for g in grads]).sum()
    total_norm = torch.sqrt(total_norm_sq)
    
    # 如果超过 max_norm，按比例缩放
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
    ) -> torch.Tensor:
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attention_weights = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kv->...qv", attention_weights, V)

class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 直接使用 Parameter（权重形状: out_features × in_features）
        self.q_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

    def forward(
        self, 
        x: Float[torch.Tensor, "... seq_len d_model"], 
        mask: torch.Tensor | None = None
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """
        Args:
            x: 输入张量 (..., seq_len, d_model)
            mask: 可选的 attention mask，如果为 None 则使用 causal mask
        Returns:
            输出张量 (..., seq_len, d_model)
        """
        # 保存原始形状
        *batch_dims, seq_len, d_model = x.shape
        
        # 步骤 1: 线性投影 Q, K, V（手动矩阵乘法）
        Q = torch.einsum('...i,oi->...o', x, self.q_proj)  # (..., seq_len, d_model)
        K = torch.einsum('...i,oi->...o', x, self.k_proj)  # (..., seq_len, d_model)
        V = torch.einsum('...i,oi->...o', x, self.v_proj)  # (..., seq_len, d_model)
        
        # 步骤 2: 分割成多头
        # reshape: (..., seq_len, d_model) → (..., seq_len, num_heads, head_dim)
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        
        # transpose: (..., seq_len, num_heads, head_dim) → (..., num_heads, seq_len, head_dim)
        Q = Q.transpose(-3, -2)  # (..., num_heads, seq_len, head_dim)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)
        
        # 如果没有提供 mask，使用 causal mask（语言模型的标准做法）
        if mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = causal_mask
        
        # 步骤 3: 计算缩放点积注意力（所有头并行计算）
        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        # (..., num_heads, seq_len, head_dim)
        
        # 步骤 4: 合并所有头
        # transpose: (..., num_heads, seq_len, head_dim) → (..., seq_len, num_heads, head_dim)
        attention_output = attention_output.transpose(-3, -2)
        
        # reshape: (..., seq_len, num_heads, head_dim) → (..., seq_len, d_model)
        attention_output = attention_output.reshape(*batch_dims, seq_len, d_model)
        
        # 步骤 5: 输出投影（手动矩阵乘法）
        output = torch.einsum('...i,oi->...o', attention_output, self.o_proj)  # (..., seq_len, d_model)
        
        return output


class MultiheadAttentionWithRoPE(torch.nn.Module):
    """MultiheadAttention with Rotary Position Embedding (RoPE)"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        # QKV 和输出投影权重
        self.q_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        
        # RoPE 模块
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device=device, dtype=dtype)

    def forward(
        self, 
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
        mask: torch.Tensor | None = None
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        *batch_dims, seq_len, d_model = x.shape
        
        # 如果没有提供 token_positions，使用默认的 0, 1, 2, ..., seq_len-1
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        
        # 步骤 1: 线性投影 Q, K, V
        Q = torch.einsum('...i,oi->...o', x, self.q_proj)
        K = torch.einsum('...i,oi->...o', x, self.k_proj)
        V = torch.einsum('...i,oi->...o', x, self.v_proj)
        
        # 步骤 2: 分割成多头
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.head_dim)
        
        # transpose: (..., seq_len, num_heads, head_dim) → (..., num_heads, seq_len, head_dim)
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)
        
        # 步骤 3: 对 Q 和 K 应用 RoPE（不对 V 应用）
        # RoPE 需要输入形状为 (..., seq_len, head_dim)
        # 当前 Q, K 形状为 (..., num_heads, seq_len, head_dim)
        # 需要对每个 head 单独应用 RoPE
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        # 步骤 4: 计算 causal mask（如果没有提供）
        if mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = causal_mask
        
        # 步骤 5: 计算缩放点积注意力
        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # 步骤 6: 合并所有头
        attention_output = attention_output.transpose(-3, -2)
        attention_output = attention_output.reshape(*batch_dims, seq_len, d_model)
        
        # 步骤 7: 输出投影
        output = torch.einsum('...i,oi->...o', attention_output, self.o_proj)
        
        return output


class TransformerBlock(torch.nn.Module):
    """Pre-norm Transformer Block"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Layer Norm 1 (before attention)
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        
        # Multi-head Attention with RoPE
        self.attn = MultiheadAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        
        # Layer Norm 2 (before FFN)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        
        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        token_positions: Int[torch.Tensor, "batch seq_len"] | None = None
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        # Pre-norm + Attention + Residual
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        
        # Pre-norm + FFN + Residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class TransformerLM(torch.nn.Module):
    """Transformer Language Model"""
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Transformer blocks
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                eps=eps,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        
        # Language model head (Linear without bias)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch_size seq_len"],
        token_positions: Int[torch.Tensor, "batch_size seq_len"] | None = None
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        batch_size, seq_len = input_ids.shape
        
        # 如果没有提供 token_positions，使用默认的 0, 1, 2, ..., seq_len-1
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        x = self.token_embeddings(input_ids)  # (batch_size, seq_len, d_model)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Language model head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits