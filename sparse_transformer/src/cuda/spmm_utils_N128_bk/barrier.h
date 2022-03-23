#ifndef BARRIER_H
#define BARRIER_H

#include <cstdint>


__device__ constexpr uint32_t StaticPow(uint32_t base, uint32_t exponent) {
  return exponent == 0 ? 1 : base * StaticPow(base, exponent - 1);
}

template <int Tile_M, int BlockWidth>
struct Barrier{
    static constexpr int kThreadsPerBlock = Tile_M * BlockWidth;
    static constexpr int kThreadsPerOutputTile = BlockWidth;
    uint32_t thread_mask = 0xffffffff;
    
    __device__ __forceinline__ Barrier(int thread_idx_y){
        if ((kThreadsPerOutputTile < 32) && (kThreadsPerOutputTile < 1)){
            constexpr uint32_t kBaseSubwarpMask = StaticPow(2, kThreadsPerOutputTile) - 1;
            thread_mask = kBaseSubwarpMask << (thread_idx_y * kThreadsPerOutputTile);
        }
    }

    __device__ __forceinline__ void Sync(){
        if (kThreadsPerOutputTile > 32){
            __syncthreads();
        } else if (kThreadsPerOutputTile > 1){
            __syncwarp(thread_mask);
        }
    }
};
#endif
