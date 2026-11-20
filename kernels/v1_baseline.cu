#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void copy_kernel(const __half* query, __half* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = query[idx];
    }
}

torch::Tensor copy_kernel_launcher(torch::Tensor query) {
    torch::Tensor output = torch::empty_like(query);
    int N = query.numel();

    dim3 threadsPerBlock(256, 1, 1);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    const __half* query_ptr = reinterpret_cast<const __half*>(query.data_ptr<at::Half>());
    __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    copy_kernel<<<numBlocks, threadsPerBlock>>>(query_ptr, output_ptr, N);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "pytorch v1 extension";
    m.def("v1", &copy_kernel_launcher);
}