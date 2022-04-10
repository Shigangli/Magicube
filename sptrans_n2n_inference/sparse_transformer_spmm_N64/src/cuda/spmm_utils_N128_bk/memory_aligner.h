#ifndef MEMORY_ALIGNER_H
#define MEMORY_ALIGNER_H

template <typename LoadType, int BlockWidth, int VecLength>
struct MemoryAligner{
    // the number of values we need to align the pointers to
    static constexpr int ValueAlignment = sizeof(LoadType) / sizeof(half) / VecLength;
    // Pre-calculated mask used to efficiently align the pointers to
    static constexpr uint32_t AlignmentMask = ~(ValueAlignment - 1);
    // The maximum number of values and indices that we could have to mask
    static constexpr int MaxValuesToMask = ValueAlignment - 1;

    // The number of masking iterations we need to perform. For most kernels,
    // this will be one and the loop in Mask Prefix should be compiled away
    static constexpr int MaskSteps = (MaxValuesToMask + BlockWidth - 1) / BlockWidth;

    //
    // Member variables
    // the row offset in the sparse matrix value & column indices budders
    int row_offset_vec_;
    // The number of nonzeros in this row of the sparse matrix
    int nonzeros_;
    // The number of values we need to mask out at the start of the first compute tile
    int values_to_mask_;

    // Constructor. Save the row offset and iniialize the masked region size
    __device__ __forceinline__ MemoryAligner(int row_offset_vec, int nonzeros){
        row_offset_vec_ = row_offset_vec;
        nonzeros_ = nonzeros;
        values_to_mask_ = row_offset_vec & (ValueAlignment - 1);
    }

    // Align the row offset 
    __device__ __forceinline__ int AlignedRowOffset(){
        return row_offset_vec_ & AlignmentMask;
    }

    __device__ __forceinline__ int AlignedNonzeros(){
        return nonzeros_ + values_to_mask_;
    }
    
    __device__ __forceinline__ void MaskPrefix(
        half* values_tile, int* column_indices_tile){
            int mask_idx = threadIdx.x;
            #pragma unroll
            for (int mask_step = 0; mask_step < MaskSteps; mask_step ++){
                if (mask_idx < values_to_mask_){
                    column_indices_tile[mask_idx] = 0;
                    #pragma unroll
                    for (int v = 0; v < VecLength; v ++){
                        values_tile[mask_idx * VecLength + v] = 0.0f;
                    }
                    mask_idx += BlockWidth;
                }
            }
        }
    
};


#endif
