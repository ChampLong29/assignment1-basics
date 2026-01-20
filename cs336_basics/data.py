import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding labels from the dataset.
    
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 计算有效的起始位置范围
    # 需要 context_length 个 token 作为输入，context_length 个 token 作为标签
    # 但标签是输入向右移动一位，所以需要 context_length + 1 个连续 token
    max_start_idx = len(dataset) - context_length - 1
    
    # 随机采样 batch_size 个起始位置
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # 构建输入和标签
    x = np.stack([dataset[i:i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])
    
    # 转换为 torch tensor 并移动到指定设备
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    return x, y
