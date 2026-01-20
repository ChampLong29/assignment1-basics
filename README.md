# CS336 Spring 2025 Assignment 1: Basics

> 🎯 **作业完成状态**: ✅ 全部通过 (46/46 tests passed, 2 skipped)

## 📋 项目概述

本项目实现了从零开始构建大语言模型 (LLM) 的核心组件，包括：

- **BPE Tokenizer**: 字节对编码分词器的训练与推理
- **Transformer 架构**: 完整的 Decoder-only Transformer 语言模型
- **训练工具**: AdamW 优化器、学习率调度、梯度裁剪等

## 🏗️ 项目结构

```
cs336_basics/
├── tokenizer.py      # BPE Tokenizer 实现
├── transformer.py    # Transformer 模型组件
├── optimizer.py      # AdamW 优化器
└── data.py          # 数据处理工具

note/
├── BPE_Implementation_Notes.md         # BPE 实现笔记
├── Tokenizer_Implementation_Summary.md  # Tokenizer 总结
└── Transformer_Implementation_Notes.md  # Transformer 实现笔记

tests/
├── adapters.py       # 测试适配器（连接实现与测试）
├── test_tokenizer.py # Tokenizer 测试
├── test_train_bpe.py # BPE 训练测试
├── test_model.py     # 模型测试
├── test_optimizer.py # 优化器测试
└── ...
```

## ✅ 已实现功能

### 1. BPE Tokenizer

| 组件 | 说明 | 状态 |
|------|------|:----:|
| `train_bpe` | BPE 词表训练算法 | ✅ |
| `Tokenizer.encode` | 文本编码为 token IDs | ✅ |
| `Tokenizer.decode` | token IDs 解码为文本 | ✅ |
| `Tokenizer.encode_iterable` | 内存高效的流式编码 | ✅ |
| Special Tokens 处理 | 支持 `<\|endoftext\|>` 等特殊标记 | ✅ |

**关键实现细节**:
- Tie-breaking: 同频 pair 按 `(bytes_a, bytes_b)` tuple 字典序选最大
- 非重叠贪心匹配：从左到右合并，避免重叠
- 增量更新 pair 计数：O(vocab_size) 而非 O(corpus_size)

### 2. Transformer 模型

| 组件 | 说明 | 状态 |
|------|------|:----:|
| `Linear` | 无偏置线性层（Xavier 初始化） | ✅ |
| `Embedding` | Token 嵌入层 | ✅ |
| `RMSNorm` | 均方根归一化 | ✅ |
| `SwiGLU` | 门控线性单元 FFN | ✅ |
| `RoPE` | 旋转位置编码 | ✅ |
| `MultiheadAttention` | 多头自注意力 | ✅ |
| `MultiheadAttentionWithRoPE` | 带 RoPE 的多头注意力 | ✅ |
| `TransformerBlock` | Pre-norm Transformer 块 | ✅ |
| `TransformerLM` | 完整语言模型 | ✅ |

**架构特点**:
- **Pre-norm** 架构（更稳定的训练）
- **RoPE** 位置编码（相对位置建模）
- **SwiGLU** 激活函数（优于 GELU/ReLU）
- **无偏置** 设计（减少参数量）

### 3. 训练工具

| 组件 | 说明 | 状态 |
|------|------|:----:|
| `softmax` | 数值稳定的 softmax | ✅ |
| `cross_entropy` | 数值稳定的交叉熵损失 | ✅ |
| `gradient_clipping` | 梯度 L2 范数裁剪 | ✅ |
| `AdamW` | 解耦权重衰减的 Adam | ✅ |
| `get_lr_cosine_schedule` | 余弦学习率调度（含 warmup） | ✅ |
| `get_batch` | 随机批次采样 | ✅ |
| `save/load_checkpoint` | 模型检查点序列化 | ✅ |

## 🚀 创新与优化

### 1. BPE 训练优化

```python
# 使用 Counter 进行高效的 pair 频率统计
pair_cnt = Counter()

# 增量更新而非全量重建
# 每次合并只更新受影响的 pair 计数
if i > 0:
    pair_cnt[(token_ids[i-1], token_a)] -= count
    pair_cnt[(token_ids[i-1], new_token_id)] += count
```

**效果**: 训练 500 vocab 在 corpus.en 上 < 1.5 秒

### 2. 数值稳定性

```python
# Log-sum-exp 技巧避免溢出
def cross_entropy(inputs, targets):
    max_vals = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_vals  # 减去最大值
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    ...
```

### 3. RoPE 预计算

```python
class RoPE(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, ...):
        # 预计算所有位置的 cos/sin 缓存
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer('cos_cached', torch.cos(angles))
        self.register_buffer('sin_cached', torch.sin(angles))
```

**效果**: 避免重复计算，支持任意 batch 维度

### 4. 高效的 einsum 操作

```python
# 使用 einsum 统一处理任意 batch 维度
Q = torch.einsum('...i,oi->...o', x, self.q_proj)  # 支持 (..., seq, d_model)
```

## 📊 测试结果

```bash
$ uv run pytest tests/ -v

tests/test_model.py::test_linear PASSED
tests/test_model.py::test_embedding PASSED
tests/test_model.py::test_swiglu PASSED
tests/test_model.py::test_scaled_dot_product_attention PASSED
tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
tests/test_model.py::test_transformer_block PASSED
tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_rmsnorm PASSED
tests/test_model.py::test_rope PASSED
tests/test_model.py::test_silu_matches_pytorch PASSED
tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED
tests/test_nn_utils.py::test_cross_entropy PASSED
tests/test_nn_utils.py::test_gradient_clipping PASSED
tests/test_optimizer.py::test_adamw PASSED
tests/test_optimizer.py::test_get_lr_cosine_schedule PASSED
tests/test_data.py::test_get_batch PASSED
tests/test_serialization.py::test_checkpointing PASSED
tests/test_tokenizer.py::test_*_matches_tiktoken PASSED (与 tiktoken 完全一致)
tests/test_train_bpe.py::test_train_bpe PASSED

======================== 46 passed, 2 skipped ========================
```

## 📝 学习笔记

详细的实现笔记位于 `note/` 目录：

- [BPE_Implementation_Notes.md](note/BPE_Implementation_Notes.md) - BPE 训练算法详解
- [Tokenizer_Implementation_Summary.md](note/Tokenizer_Implementation_Summary.md) - Tokenizer 编码解码
- [Transformer_Implementation_Notes.md](note/Transformer_Implementation_Notes.md) - Transformer 组件实现

## 🔧 Setup

### 环境配置

```bash
# 安装 uv (推荐)
pip install uv
# 或
brew install uv

# 运行代码
uv run <python_file>

# 运行测试
uv run pytest
```

### 下载数据

```bash
mkdir -p data && cd data

# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText 样本
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_*.txt.gz

cd ..
```

## 🎓 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - AdamW
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - BPE

---

*Stanford CS336: Language Modeling from Scratch (Spring 2025)*
