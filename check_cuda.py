"""Quick check that CUDA is available and working."""

import torch

print(f"PyTorch version:  {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:     {torch.version.cuda}")
    print(f"Device count:     {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        mem = torch.cuda.get_device_properties(i).total_memory
        print(f"      Memory: {mem / 1024**3:.1f} GB")

    # Smoke test: allocate a tensor and do a matmul on GPU
    device = torch.device("cuda")
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    c = a @ b
    print(f"\nSmoke test passed: matmul result shape {c.shape} on {c.device}")
else:
    print("\nCUDA is NOT available. Training will run on CPU.")
