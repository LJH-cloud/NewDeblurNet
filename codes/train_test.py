def print_cuda_mem(stage=""):
    print(f"[{stage}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[{stage}] Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
if __name__ == "__main__":
    import torch
    print_cuda_mem("Initial")