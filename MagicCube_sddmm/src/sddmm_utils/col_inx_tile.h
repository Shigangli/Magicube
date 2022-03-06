#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sddmm{
    struct sddmm_col_idx_tile{

        const int lane_id_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        // shared memory tile for sparse marix values
        int * column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ sddmm_col_idx_tile(
            int offset, int , int warp_id,
            const VecType * __restrict__ values,
            const int * __restrict__ column_idxs,
            int * column_idxs_tile):
            in_warp_tid_(in_warp_tid),
            warp_id_(warp_id),
            values_(reinterpret_cast<const int *>(values + row_offset_vec) + in_warp_tid),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec) + in_warp_tid),
            values_tile_base_(reinterpret_cast<int *>(values_tile) + in_warp_tid),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + in_warp_tid){}
        
        // Load
        __device__ __forceinline__ void Load(){
            int * values_tile = values_tile_base_ + (step % 2) * ValuesBlockWidth;
            int * column_idxs_tile = column_idxs_tile_base_ + (step % 2) * BlockWidth;

	    if(warp_id_ == 0 && in_warp_tid_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if(in_warp_tid_ < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            values_ += ValuesBlockWidth;
            column_idxs_ += BlockWidth;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(){
            int * values_tile = values_tile_base_;
            int * column_idxs_tile = column_idxs_tile_base_;
	    if(warp_id_ == 0 && in_warp_tid_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if(in_warp_tid_ < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }
    };
}
#endif
