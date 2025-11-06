import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Example tensor operation on GPU
    a = torch.randn((1000, 1000), device=device)
    b = torch.randn((1000, 1000), device=device)
    c = a + b
    print("Tensor operation on GPU complete.")
else:
    print("CUDA is not available. Using CPU.")