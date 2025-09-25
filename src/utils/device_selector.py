import logging
import torch
import numpy as np
from src.utils.args_parser import args


def get_free_gpu(gpu_exclude_list=[]):
    """
    Return the GPU index with the most free memory, excluding those in gpu_exclude_list.
    Works across Linux/Windows/macOS without relying on nvidia-smi.
    """
    if not torch.cuda.is_available():
        return None

    free_memories = []
    for i in range(torch.cuda.device_count()):
        if i in gpu_exclude_list:
            free_memories.append(-1)  # mark excluded GPU
            continue

        try:
            stats = torch.cuda.mem_get_info(i)  # (free, total) in bytes
            free_memories.append(stats[0])
        except Exception as e:
            logging.warning(f"Could not query memory for GPU {i}: {e}")
            free_memories.append(-1)

    if all(mem == -1 for mem in free_memories):
        return None

    return int(np.argmax(free_memories))


def get_free_device_name(gpu_exclude_list=[]):
    """
    Return a torch device string ('cuda:X' or 'cpu').
    Picks the GPU with the most free memory, unless excluded.
    """
    if torch.cuda.is_available() and getattr(args, "gpu", True):
        gpu = get_free_gpu(gpu_exclude_list=gpu_exclude_list)
        if gpu is not None:
            logging.info(f"Using GPU: {gpu}")
            return f"cuda:{gpu}"
        else:
            logging.info("No suitable GPU found, falling back to CPU.")
            return "cpu"
    else:
        logging.info("CUDA not available, using CPU.")
        return "cpu"
