// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/segmented_lora_gemm_kernel.h"
#include "paddle/phi/kernels/gpu/segmented_lora_gemm_impl.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/common/memory_utils.h"
#include "glog/logging.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"


namespace phi {

template <typename T>
struct cutlass_dtype_traits {
  using type = T;
  using paddle_out_type = T;
  using out_type = T;
  using accum_type = float;
};

template <>
struct cutlass_dtype_traits<phi::dtype::float16> {
  using type = cutlass::half_t;
  using paddle_out_type = phi::dtype::float16;
  using out_type = cutlass::half_t;
  using accum_type = float;
};

template <>
struct cutlass_dtype_traits<phi::dtype::bfloat16> {
  using type = cutlass::bfloat16_t;
  using paddle_out_type = phi::dtype::bfloat16;
  using out_type = cutlass::bfloat16_t;
  using accum_type = float;
};

template <>
struct cutlass_dtype_traits<int8_t> {
  using type = int8_t;
  using paddle_out_type = int32_t;
  using out_type = int32_t;
  using accum_type = int32_t;
};

template <typename T, typename Context>
void SegmentedLoRAGemmKernel(const Context& dev_ctx,
                             const DenseTensor& x, // [token_num, d_in]
                             const DenseTensor& lora_weights, // [lora_num, d_out, d_in]
                             const DenseTensor& lora_dequant_scale,
                             const DenseTensor& lora_quant_scale,
                             const DenseTensor& w_offsets,
                             const phi::DenseTensor& padding_offsets,
                             const phi::DenseTensor& cum_offsets,
                             const DenseTensor& seq_lens_this_time,
                             const DenseTensor& seq_lens_encoder,
                             const DenseTensor& seq_lens_decoder,
                             const DenseTensor& cu_seqlens_q,
                             const DenseTensor& cu_seqlens_k,
                             DenseTensor* out) {
  using CUTLASS_T = typename cutlass_dtype_traits<T>::type;
  using PADDLE_OUT_T = typename cutlass_dtype_traits<T>::paddle_out_type;
  using CUTLASS_OUT_T = typename cutlass_dtype_traits<T>::out_type;
  using CUTLASS_ACCUM_T = typename cutlass_dtype_traits<T>::accum_type;
  auto stream = dev_ctx.stream();
  auto x_dims = x.dims();
  auto lora_dims = lora_weights.dims();
  const int token_num = x_dims[0];
  const int d_in = x_dims[1];
  const int lora_num = lora_dims[0];
  const int d_out = 64; // lora_dims[1];
  const int num_problems = seq_lens_this_time.dims()[0];
  const T* x_data = x.data<T>();
  const T* w_data = lora_weights.data<T>();
  const int* s_data = cu_seqlens_q.data<int32_t>();
  const uint32_t num_layer = 1;
  const uint32_t layer_id = 0;
  // VLOG(0) << "num_problems: " << num_problems;

  dev_ctx.template Alloc<PADDLE_OUT_T>(out);
  PADDLE_OUT_T* out_data = out->data<PADDLE_OUT_T>();
  // ---------------------------------
  constexpr uint32_t num_warps = 4;
  constexpr uint32_t num_stages = 2;
  constexpr uint32_t num_k_frags_per_stage = 8;
  constexpr uint32_t num_blocks_n = d_out / 16;
  uint32_t smem = num_stages * sizeof(T) * num_k_frags_per_stage * 16 * 16 *
                  (num_warps + num_blocks_n);
  bool use_cooperative = true;
  auto cooperative_kernel =
      sgmm::sgmm_shrink<true, T, int, num_warps, d_out>;
  auto kernel = sgmm::sgmm_shrink<false, T, int, num_warps, d_out>;
  uint32_t chunk_size = 256;
  uint32_t num_chunks = (d_in + chunk_size - 1) / chunk_size;
  dim3 nthrs(32, num_warps); // [32, 4]
  dim3 nblks(num_chunks, num_problems); //[1, num_problems]
  // ---------------------------------
  if (d_in < d_out) { // k is small (16)
    // Expand
  } else { // k is large, N is small
    // Shrink
    VLOG(0) << "Shrink";
    phi::DenseTensor workspace;
    workspace.Resize(phi::make_ddim({static_cast<int64_t>(8 * 1024 * 1024)}));
    dev_ctx.template Alloc<uint8_t>(&workspace);
    uint8_t *tmp = workspace.data<uint8_t>();
    
    void* args[] = {(void*)&out_data,    (void*)&x_data,         (void*)&w_data,
                  (void*)&s_data,    (void*)&tmp,       (void*)&num_problems,
                  (void*)&d_in, (void*)&w_offsets,  (void*)&chunk_size,
                  (void*)&num_layer, (void*)&layer_id};
    cudaError_t status;
    VLOG(0) << "launch kernel";
    if (use_cooperative) {
      if (smem > 46 * 1024) {
        cudaFuncSetAttribute(cooperative_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      }
      status = cudaLaunchCooperativeKernel((void*)cooperative_kernel, nblks,
                                          nthrs, args, smem, stream);
    } else {
      if (smem > 46 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem);
      }
      status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem, stream);
      
    }
    VLOG(1) << "kernel status: " << status;
  }
}

} // namespace phi

PD_REGISTER_KERNEL(segmented_lora_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::SegmentedLoRAGemmKernel,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}