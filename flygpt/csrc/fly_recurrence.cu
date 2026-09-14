// Fused CUDA kernels for the FlyRNN recurrence (plan.md §7, §8).
//
//   proposal_i = tanh( sum_j W_ij h_j + drive_i + bias_i )        W_ij = vals_e / sqrt(in_degree_i), CSR rows = dst
//   h_i        = (1 - leak_i) h_i + leak_i proposal_i
//
// Layout: neuron states are [N, B] fp32 so one neuron's batch row is a contiguous 128-byte line and one
// warp (32 lanes = 32 batch columns) reads it in a single transaction. One warp per destination row in
// the forward pass, one warp per source row for the transposed product in the backward pass, one warp
// per edge for the edge-value gradient. The whole T×microsteps window runs inside one host loop so a
// training step launches ~4 kernels per microstep instead of ~100.
//
// The math is identical to FlyRNN.step / dense_reference_step; tests/test_kernels.py checks logits and
// every gradient against the PyTorch sparse path.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace {

constexpr int WARP = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---- forward: WPR warps per destination neuron, each warp a slice of the row's edges -----------------
// s_out[i,b] = (1-leak_i) s_in[i,b] + leak_i tanh( sum_e vals[e] s_in[col[e],b] + drive[in_pos[i],b] + bias_i )
template <int WPR>
__global__ void fwd_row_kernel(const float* __restrict__ vals, const int* __restrict__ col, const int* __restrict__ crow,
                               const float* __restrict__ s_in, float* __restrict__ s_out, float* __restrict__ p_out,
                               const float* __restrict__ drive, const int* __restrict__ in_pos,
                               const float* __restrict__ leak, const float* __restrict__ bias, int N, int B) {
    __shared__ float red[WPR][WARP];
    const int i = blockIdx.x;
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int start = crow[i], end = crow[i + 1];
    const int ip = in_pos[i];
    const float lk = leak[i], bi = bias[i];
    for (int b0 = 0; b0 < B; b0 += WARP) {
        const int b = b0 + lane;
        const bool ok = b < B;
        float acc = 0.f;
        for (int e = start + w; e < end; e += WPR) {                // vals/col loads are warp-uniform (broadcast)
            const float v = vals[e];
            const int c = col[e];
            acc += ok ? v * s_in[(size_t)c * B + b] : 0.f;
        }
        red[w][lane] = acc;
        __syncthreads();
        if (w == 0 && ok) {
#pragma unroll
            for (int k = 1; k < WPR; ++k) acc += red[k][lane];
            const float d = ip >= 0 ? drive[(size_t)ip * B + b] : 0.f;
            const float p = tanhf(acc + d + bi);
            const size_t idx = (size_t)i * B + b;
            s_out[idx] = (1.f - lk) * s_in[idx] + lk * p;
            p_out[idx] = p;
        }
        __syncthreads();
    }
}

// ---- backward 1: per neuron, elementwise grads + row reductions ---------------------------------------
// g_p = g_s * leak * (1 - p^2);  g_bias_i += sum_b g_p;  g_leak_i += sum_b g_s (p - s_old);  g_drive[in_pos] += g_p
__global__ void bwd_row_kernel(const float* __restrict__ g_s, const float* __restrict__ s_old, const float* __restrict__ p,
                               const float* __restrict__ leak, const int* __restrict__ in_pos,
                               float* __restrict__ g_p, float* __restrict__ g_bias, float* __restrict__ g_leak,
                               float* __restrict__ g_drive, int N, int B) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP;
    if (warp_id >= N) return;
    const int i = warp_id;
    const float lk = leak[i];
    const int ip = in_pos[i];
    float sb = 0.f, sl = 0.f;
    for (int b0 = 0; b0 < B; b0 += WARP) {
        const int b = b0 + lane;
        if (b < B) {
            const size_t idx = (size_t)i * B + b;
            const float gs = g_s[idx], pp = p[idx];
            const float gp = gs * lk * (1.f - pp * pp);
            g_p[idx] = gp;
            sb += gp;
            sl += gs * (pp - s_old[idx]);
            if (ip >= 0) g_drive[(size_t)ip * B + b] += gp;         // one warp owns row i: no race
        }
    }
    sb = warp_sum(sb); sl = warp_sum(sl);
    if (lane == 0) { g_bias[i] += sb; g_leak[i] += sl; }
}

// ---- backward 2: WPR warps per source neuron, transposed CSR product ----------------------------------
// g_s_old[j,b] = (1-leak_j) g_s[j,b] + sum_{e: src_e = j} vals[e] g_p[dst_e, b]
template <int WPR>
__global__ void bwd_state_kernel(const float* __restrict__ vals, const int* __restrict__ permT, const int* __restrict__ colT,
                                 const int* __restrict__ crowT, const float* __restrict__ g_p, const float* __restrict__ g_s,
                                 const float* __restrict__ leak, float* __restrict__ g_s_old, int N, int B) {
    __shared__ float red[WPR][WARP];
    const int j = blockIdx.x;
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int start = crowT[j], end = crowT[j + 1];
    const float lk = leak[j];
    for (int b0 = 0; b0 < B; b0 += WARP) {
        const int b = b0 + lane;
        const bool ok = b < B;
        float acc = 0.f;
        for (int e = start + w; e < end; e += WPR) {
            const float v = vals[permT[e]];
            const int c = colT[e];
            acc += ok ? v * g_p[(size_t)c * B + b] : 0.f;
        }
        red[w][lane] = acc;
        __syncthreads();
        if (w == 0 && ok) {
#pragma unroll
            for (int k = 1; k < WPR; ++k) acc += red[k][lane];
            const size_t idx = (size_t)j * B + b;
            g_s_old[idx] = (1.f - lk) * g_s[idx] + acc;
        }
        __syncthreads();
    }
}

// ---- backward 3: per edge, dot product over the batch ---------------------------------------------------
// g_vals[e] += sum_b g_p[dst_e, b] s_old[src_e, b]
__global__ void bwd_vals_kernel(const int* __restrict__ src, const int* __restrict__ dst, const float* __restrict__ g_p,
                                const float* __restrict__ s_old, float* __restrict__ g_vals, int E, int B) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP;
    if (warp_id >= E) return;
    const int e = warp_id;
    const size_t rd = (size_t)dst[e] * B, rs = (size_t)src[e] * B;
    float acc = 0.f;
    for (int b = lane; b < B; b += WARP) acc += g_p[rd + b] * s_old[rs + b];
    acc = warp_sum(acc);
    if (lane == 0) g_vals[e] += acc;
}

inline dim3 warps_grid(int64_t n_warps, int threads = 256) {
    return dim3((unsigned)((n_warps * WARP + threads - 1) / threads));
}

}  // namespace

// Runs T characters × M microsteps. Returns (states [K+1, N, B], props [K, N, B]); states[0] = state0.
std::vector<torch::Tensor> fly_forward(torch::Tensor vals, torch::Tensor col, torch::Tensor crow, torch::Tensor in_pos,
                                       torch::Tensor leak, torch::Tensor bias, torch::Tensor drives, torch::Tensor state0,
                                       int64_t microsteps) {
    const at::cuda::CUDAGuard guard(vals.device());
    const int N = (int)leak.size(0), B = (int)state0.size(1), T = (int)drives.size(0), K = T * (int)microsteps;
    auto states = torch::empty({K + 1, N, B}, vals.options());
    auto props = torch::empty({K, N, B}, vals.options());
    states[0].copy_(state0);
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WPR = 16;
    for (int k = 0; k < K; ++k) {
        fwd_row_kernel<WPR><<<N, WPR * WARP, 0, stream>>>(
            vals.data_ptr<float>(), col.data_ptr<int>(), crow.data_ptr<int>(),
            states[k].data_ptr<float>(), states[k + 1].data_ptr<float>(), props[k].data_ptr<float>(),
            drives[k / microsteps].data_ptr<float>(), in_pos.data_ptr<int>(),
            leak.data_ptr<float>(), bias.data_ptr<float>(), N, B);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {states, props};
}

// g_out: [T, N, B] gradient wrt the state after each character. Returns (g_vals, g_leak, g_bias, g_drives, g_state0).
std::vector<torch::Tensor> fly_backward(torch::Tensor vals, torch::Tensor src, torch::Tensor dst, torch::Tensor permT,
                                        torch::Tensor colT, torch::Tensor crowT, torch::Tensor in_pos, torch::Tensor leak,
                                        torch::Tensor states, torch::Tensor props, torch::Tensor g_out,
                                        int64_t microsteps, int64_t n_in) {
    const at::cuda::CUDAGuard guard(vals.device());
    const int N = (int)leak.size(0), B = (int)states.size(2), K = (int)props.size(0), T = K / (int)microsteps;
    const int E = (int)vals.size(0);
    auto g_vals = torch::zeros_like(vals);
    auto g_bias = torch::zeros({N}, vals.options());
    auto g_leak = torch::zeros({N}, vals.options());
    auto g_drives = torch::zeros({T, n_in, B}, vals.options());
    auto g_p = torch::empty({N, B}, vals.options());
    auto g_s = torch::zeros({N, B}, vals.options());
    auto g_s_old = torch::empty({N, B}, vals.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WPR = 16;
    const dim3 grid_n = warps_grid(N), grid_e = warps_grid(E);
    for (int k = K - 1; k >= 0; --k) {
        const int t = k / (int)microsteps;
        if ((k + 1) % microsteps == 0) g_s.add_(g_out[t]);
        bwd_row_kernel<<<grid_n, 256, 0, stream>>>(
            g_s.data_ptr<float>(), states[k].data_ptr<float>(), props[k].data_ptr<float>(), leak.data_ptr<float>(),
            in_pos.data_ptr<int>(), g_p.data_ptr<float>(), g_bias.data_ptr<float>(), g_leak.data_ptr<float>(),
            g_drives[t].data_ptr<float>(), N, B);
        bwd_state_kernel<WPR><<<N, WPR * WARP, 0, stream>>>(
            vals.data_ptr<float>(), permT.data_ptr<int>(), colT.data_ptr<int>(), crowT.data_ptr<int>(),
            g_p.data_ptr<float>(), g_s.data_ptr<float>(), leak.data_ptr<float>(), g_s_old.data_ptr<float>(), N, B);
        bwd_vals_kernel<<<grid_e, 256, 0, stream>>>(
            src.data_ptr<int>(), dst.data_ptr<int>(), g_p.data_ptr<float>(), states[k].data_ptr<float>(),
            g_vals.data_ptr<float>(), E, B);
        std::swap(g_s, g_s_old);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_vals, g_leak, g_bias, g_drives, g_s};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fly_forward, "FlyRNN fused recurrence forward (CUDA)");
    m.def("backward", &fly_backward, "FlyRNN fused recurrence backward (CUDA)");
}
