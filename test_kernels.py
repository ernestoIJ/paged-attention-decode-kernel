import torch
from torch.utils.cpp_extension import load
from test_setup import create_data

head_dim = 128
num_kv_heads = 8
num_query_heads = 32
batch_size = 64
seq_len = 4096
page_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

data = create_data(head_dim=head_dim, num_kv_heads=num_kv_heads, num_query_heads=num_query_heads, batch_size=batch_size, seq_len=seq_len, page_size=page_size, device=device)
module = load(name="v1", sources=["kernels/v1_baseline.cu"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=True)
res = module.v1(data["query"])

if torch.equal(res, data["query"]):
    print("PASS")
else:
    print("FAIL")