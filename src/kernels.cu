#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "../tester/utils.h"

#define WARP_SIZE 32

// ==========================================
// 精度工具：强制使用 Double 处理 Float 输入
// ==========================================
template<typename T>
struct PrecisionUtil;

// Float 输入 -> 内部使用 double 计算
template<>
struct PrecisionUtil<float> {
    using Type = double;
    static __device__ __forceinline__ double to_prec(float val) { return (double)val; }
    static __device__ __forceinline__ float from_prec(double val) { return (float)val; }
};

// Half 输入 -> 内部使用 float 计算 (Half 精度要求不高，float 足够且快)
template<>
struct PrecisionUtil<half> {
    using Type = float;
    static __device__ __forceinline__ float to_prec(half val) { return __half2float(val); }
    static __device__ __forceinline__ half from_prec(float val) { return __float2half(val); }
};

// ==========================================
// Part 1: Trace (标准实现)
// ==========================================
template <typename T>
__global__ void trace_kernel(const T* input, T* output, int rows, int cols, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        size_t flat_idx = (size_t)idx * cols + idx;
        atomicAdd(output, input[flat_idx]);
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return T(0);
    T* d_input = nullptr; T* d_output = nullptr;
    size_t size_bytes = rows * cols * sizeof(T);
    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, sizeof(T));
    cudaMemset(d_output, 0, sizeof(T));
    cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice);
    int limit = (rows < cols) ? rows : cols;
    int threads = 256;
    int blocks = (limit + threads - 1) / threads;
    trace_kernel<<<blocks, threads>>>(d_input, d_output, rows, cols, limit);
    T h_res;
    cudaMemcpy(&h_res, d_output, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output);
    return h_res;
}

// ==========================================
// Part 2: Flash Attention (Tiled 1-Pass)
// ==========================================

// Warp Reduction Sum (适配 Double/Float)
template<typename AccT>
__inline__ __device__ AccT warp_reduce_sum(AccT val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        if (sizeof(AccT) == 8) { // Double
            unsigned long long int_val = *reinterpret_cast<unsigned long long*>(&val);
            unsigned long long shfl_val = __shfl_xor_sync(0xffffffff, int_val, mask);
            val += *reinterpret_cast<double*>(&shfl_val);
        } else { // Float
            val += __shfl_xor_sync(0xffffffff, (float)val, mask);
        }
    }
    return val;
}

template <typename T>
__global__ void flash_attention_tiled_1pass_kernel(
    const T* __restrict__ Q, 
    const T* __restrict__ K, 
    const T* __restrict__ V, 
    T* __restrict__ O,
    int batch_size,
    int tgt_seq_len, int src_seq_len, 
    int query_heads, int kv_heads, int head_dim, 
    double scale_val, // 始终传 double 以保证精度
    bool is_causal) 
{
    using AccT = typename PrecisionUtil<T>::Type;

    // Grid: [Batch * Heads * Tgt_Seq] (Flattened)
    // Block: 32 Threads (1 Warp)
    int global_q_idx = blockIdx.x;
    
    // Index Mapping
    int tgt_idx = global_q_idx % tgt_seq_len;
    int rem = global_q_idx / tgt_seq_len;
    int head_idx = rem % query_heads;
    int batch_idx = rem / query_heads;

    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;

    // Offsets
    int kv_head_idx = head_idx / (query_heads / kv_heads);
    size_t batch_offset_q = (size_t)batch_idx * tgt_seq_len * query_heads * head_dim;
    size_t batch_offset_kv = (size_t)batch_idx * src_seq_len * kv_heads * head_dim;
    
    size_t q_offset = batch_offset_q + (size_t)tgt_idx * query_heads * head_dim + head_idx * head_dim;
    size_t k_base = batch_offset_kv + kv_head_idx * head_dim;
    size_t v_base = k_base;

    // 1. Load Q into Registers
    // 支持 head_dim 高达 256 (8个寄存器)
    AccT q_reg[8]; 
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int d = i * 32 + tid;
        if (d < head_dim) q_reg[i] = PrecisionUtil<T>::to_prec(Q[q_offset + d]);
        else q_reg[i] = (AccT)0.0;
    }

    // 2. Global Accumulators
    AccT m_global = -INFINITY;
    AccT l_global = (AccT)0.0;
    AccT acc_global[8];
    #pragma unroll
    for(int i=0; i<8; ++i) acc_global[i] = (AccT)0.0;

    // 3. Tiled Loop
    const int TILE_SIZE = 32; // 每个 Tile 处理 32 个 Key
    int loop_end = src_seq_len;
    if (is_causal) {
        loop_end = (tgt_idx + 1 < src_seq_len) ? (tgt_idx + 1) : src_seq_len;
    }

    for (int src_base = 0; src_base < loop_end; src_base += TILE_SIZE) {
        int current_tile_limit = (src_base + TILE_SIZE < loop_end) ? (src_base + TILE_SIZE) : loop_end;
        
        // --- Pass A: Find Tile Max ---
        AccT m_tile = -INFINITY;
        
        for (int k = src_base; k < current_tile_limit; ++k) {
            size_t k_ptr = k_base + (size_t)k * kv_heads * head_dim;
            AccT dot = (AccT)0.0;
            
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int d = i * 32 + tid;
                if (d < head_dim) {
                    AccT k_val = PrecisionUtil<T>::to_prec(K[k_ptr + d]);
                    // dot += q * k
                    dot += q_reg[i] * k_val;
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= (AccT)scale_val;
            
            if (dot > m_tile) m_tile = dot;
        }

        // --- Pass B: Accumulate Tile (Recompute Dot to save registers) ---
        AccT l_tile = (AccT)0.0;
        AccT acc_tile[8];
        #pragma unroll
        for(int i=0; i<8; ++i) acc_tile[i] = (AccT)0.0;

        for (int k = src_base; k < current_tile_limit; ++k) {
            size_t k_ptr = k_base + (size_t)k * kv_heads * head_dim;
            AccT dot = (AccT)0.0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int d = i * 32 + tid;
                if (d < head_dim) {
                    AccT k_val = PrecisionUtil<T>::to_prec(K[k_ptr + d]);
                    dot += q_reg[i] * k_val;
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= (AccT)scale_val;

            AccT p = (AccT)0.0;
            if (m_tile > -1e30) {
                p = exp(dot - m_tile);
            }
            l_tile += p;

            size_t v_ptr = v_base + (size_t)k * kv_heads * head_dim;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int d = i * 32 + tid;
                if (d < head_dim) {
                    AccT v_val = PrecisionUtil<T>::to_prec(V[v_ptr + d]);
                    acc_tile[i] += p * v_val;
                }
            }
        }

        // --- Pass C: Merge Tile to Global (Rescaling happens here) ---
        AccT m_new = (m_global > m_tile) ? m_global : m_tile;
        AccT f_global = (AccT)1.0;
        AccT f_tile = (AccT)0.0;

        if (m_new > -1e30) {
            if (m_global > -1e30) f_global = exp(m_global - m_new);
            else f_global = (AccT)0.0; // Global was empty/masked

            if (m_tile > -1e30) f_tile = exp(m_tile - m_new);
        }

        l_global = l_global * f_global + l_tile * f_tile;
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc_global[i] = acc_global[i] * f_global + acc_tile[i] * f_tile;
        }
        m_global = m_new;
    }

    // 4. Final Write
    AccT inv_l = (l_global > 1e-9) ? ((AccT)1.0 / l_global) : (AccT)0.0;
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int d = i * 32 + tid;
        if (d < head_dim) {
            AccT out = acc_global[i] * inv_l;
            O[q_offset + d] = PrecisionUtil<T>::from_prec(out);
        }
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {        
    
    size_t size_o = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    h_o.resize(size_o);

    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, h_q.size()*sizeof(T));
    cudaMalloc(&d_k, h_k.size()*sizeof(T));
    cudaMalloc(&d_v, h_v.size()*sizeof(T));
    cudaMalloc(&d_o, h_o.size()*sizeof(T));

    cudaMemcpy(d_q, h_q.data(), h_q.size()*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), h_k.size()*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), h_v.size()*sizeof(T), cudaMemcpyHostToDevice);

    // Launch Config
    // One Warp (32 threads) per Query Token
    int total_tasks = batch_size * query_heads * target_seq_len;
    dim3 grid(total_tasks);
    dim3 block(32); 

    double scale_d = 1.0 / sqrt((double)head_dim);

    flash_attention_tiled_1pass_kernel<T><<<grid, block>>>(d_q, d_k, d_v, d_o, 
                                                           batch_size, 
                                                           target_seq_len, src_seq_len, 
                                                           query_heads, kv_heads, head_dim, 
                                                           scale_d, is_causal);

    cudaMemcpy(h_o.data(), d_o, size_o*sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}

// Explicit Instantiations
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, std::vector<float>&, int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&, const std::vector<half>&, std::vector<half>&, int, int, int, int, int, int, bool);