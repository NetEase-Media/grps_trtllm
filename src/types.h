// Types definitions.

#pragma once

#include <cstdint>

namespace netease::grps {
typedef uint16_t bf16;

union {
  float f;
  uint32_t i;
} typedef FloatBits;

static inline bf16 Float32ToBfloat16(float value) {
  FloatBits float_bits;
  float_bits.f = value;
  return static_cast<bf16>(float_bits.i >> 16);
}

static inline float Bfloat16ToFloat32(bf16 value) {
  FloatBits float_bits;
  float_bits.i = static_cast<uint32_t>(value) << 16;
  return float_bits.f;
}
} // namespace netease::grps