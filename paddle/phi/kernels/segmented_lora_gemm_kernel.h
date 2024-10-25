
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

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void SegmentedLoRAGemmKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& lora_weights,
                             const DenseTensor& lora_dequant_scale,
                             const DenseTensor& lora_quant_scale,
                             const DenseTensor& w_offsets,
                             const DenseTensor& padding_offsets,
                             const DenseTensor& cum_offsets,
                             const DenseTensor& seq_lens_this_time,
                             const DenseTensor& seq_lens_encoder,
                             const DenseTensor& seq_lens_decoder,
                             const DenseTensor& cu_seqlens_q,
                             const DenseTensor& cu_seqlens_k,
                             DenseTensor* out);
}  // namespace phi
