#ifndef SPMM_DENSE_TILE_H
#define SPMM_DENSE_TILE_H

#include <cuda_fp16.h>

namespace spmm {
    template <typename LoadType, typename VecType, int Tile_K, int Tile_N, int BlockWidth, int VecLength>
    struct DenseTile {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsK = Tile_K / BlockWidth / kValuesPerLoad_;
        static constexpr int kScalarThreadItemsK = Tile_K / BlockWidth;
        static constexpr int kResidueUnroll = 4;
        static constexpr int kResidueOuterLimit_ = Tile_K / kResidueUnroll;
        static constexpr int kResidueInnerLimit_ = kResidueUnroll;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offsets in shared memory
        const int* row_offsets_base_;
        // The register file fragment to load the dense values into.
        LoadType *matrix_fragment_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ DenseTile(
            int rhs_columns, int offset, int thread_idx_x,
            const half* __restrict__ matrix,
            const int *row_offsets,
            half * matrix_fragment):
            rhs_columns_(rhs_columns * sizeof(half)),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset) + thread_idx_x),
            row_offsets_base_(row_offsets),
            matrix_fragment_(reinterpret_cast<LoadType *>(matrix_fragment)){}

        // Load
        __device__ __forceinline__ void Load(){
            const int *row_offsets = row_offsets_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < Tile_K; n_item_idx ++){
                const LoadType* matrix = matrix_base_ + *(row_offsets);

                #pragma unroll
                for (int k_item_idx=0; k_item_idx < kThreadItemsK; k_item_idx ++){
                    int fragment_offset = n_item_idx * kThreadItemsK + k_item_idx;
                    matrix_fragment_[fragment_offset] = __ldg(matrix);
                    matrix += 32;
                }
                row_offsets ++;
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoadAndCompute(
            int residue, const half *lhs_tile, float *output_fragment){
            
            const VecType * lhs_tile_v = reinterpret_cast<const VecType *>(lhs_tile);
            
            const int *row_offsets = row_offsets_base_;

            #pragma unroll
            for (int n_outer_idx = 0; n_outer_idx < kResidueOuterLimit_; n_outer_idx ++){
                if (residue <= 0) return;

                #pragma unroll
                for (int n_inner_idx = 0; n_inner_idx < kResidueInnerLimit_; n_inner_idx ++){
                    const int n_item_idx = n_inner_idx + n_outer_idx * kResidueInnerLimit_;
                    
                    int indices = *(row_offsets);
                    half lhs_value[VecLength];
                    VecType * lhs_value_v = reinterpret_cast<VecType *>(lhs_value);
                    *(lhs_value_v) = *(lhs_tile_v + n_item_idx);

                    const LoadType* matrix = matrix_base_ + indices;

                    #pragma unroll
                    for (int k_item_idx = 0; k_item_idx < kThreadItemsK; k_item_idx ++){
                        half rhs_values[kValuesPerLoad_];
                        LoadType* rhs_values_t = reinterpret_cast<LoadType* >(rhs_values);
                        *(rhs_values_t) = __ldg(matrix);
                        #pragma unroll
                        for (int v = 0; v < kValuesPerLoad_; v++){
                            float* outputs = output_fragment + k_item_idx * kValuesPerLoad_ + v;
                            half rhs_value = rhs_values[v];
                            #pragma unroll
                            for (int vl = 0; vl < VecLength; vl ++){
                                *(outputs + vl * kScalarThreadItemsK) += __half2float(lhs_value[vl] * rhs_value);
                            }
                        }
                        matrix += BlockWidth;
                    }

                    row_offsets ++;
                }

                residue -= kResidueInnerLimit_;
            }
            asm("");
        }
    };

    template <typename LoadType, typename VecType, int Tile_K, int Tile_N, int BlockWidth, int VecLength>
    struct DenseTile1D {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsK = Tile_K / BlockWidth / kValuesPerLoad_;
        static constexpr int kScalarThreadItemsK = Tile_K / BlockWidth;
        static constexpr int kResidueUnroll = 4;
        static constexpr int kResidueOuterLimit_ = Tile_K / kResidueUnroll;
        static constexpr int kResidueInnerLimit_ = kResidueUnroll;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offsets in shared memory
        const int* row_offsets_base_;
        // The register file fragment to load the dense values into.
        LoadType *matrix_fragment_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ DenseTile1D(
            int rhs_columns, int offset, int thread_idx_x,
            const half* __restrict__ matrix,
            const int *row_offsets,
            half * matrix_fragment):
            rhs_columns_(rhs_columns * sizeof(half)),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset) + thread_idx_x),
            row_offsets_base_(row_offsets),
            matrix_fragment_(reinterpret_cast<LoadType *>(matrix_fragment)){}

        // Load
        __device__ __forceinline__ void Load(){
            const int *row_offsets = row_offsets_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < Tile_K; n_item_idx ++){
                const LoadType* matrix = matrix_base_ + *(row_offsets);

                #pragma unroll
                for (int k_item_idx=0; k_item_idx < kThreadItemsK; k_item_idx ++){
                    int fragment_offset = n_item_idx * kThreadItemsK + k_item_idx;
                    matrix_fragment_[fragment_offset] = __ldg(matrix);
                    matrix += 32;
                }
                row_offsets ++;
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoadAndCompute(
            int residue, VecType *lhs_tile, float *output_fragment){
            
            const VecType * lhs_tile_v = reinterpret_cast<const VecType *>(lhs_tile);
            
            const int *row_offsets = row_offsets_base_;

            #pragma unroll
            for (int n_outer_idx = 0; n_outer_idx < kResidueOuterLimit_; n_outer_idx ++){
                if (residue <= 0) return;

                #pragma unroll
                for (int n_inner_idx = 0; n_inner_idx < kResidueInnerLimit_; n_inner_idx ++){
                    const int n_item_idx = n_inner_idx + n_outer_idx * kResidueInnerLimit_;
                    
                    int indices = *(row_offsets);
                    half lhs_value[VecLength];
                    VecType * lhs_value_v = reinterpret_cast<VecType *>(lhs_value);
                    *(lhs_value_v) = *(lhs_tile_v + n_item_idx);

                    const LoadType* matrix = matrix_base_ + indices;

                    #pragma unroll
                    for (int k_item_idx = 0; k_item_idx < kThreadItemsK; k_item_idx ++){
                        half rhs_values[kValuesPerLoad_];
                        LoadType* rhs_values_t = reinterpret_cast<LoadType* >(rhs_values);
                        *(rhs_values_t) = __ldg(matrix);
                        #pragma unroll
                        for (int v = 0; v < kValuesPerLoad_; v++){
                            float* outputs = output_fragment + k_item_idx * kValuesPerLoad_ + v;
                            half rhs_value = rhs_values[v];
                            #pragma unroll
                            for (int vl = 0; vl < VecLength; vl ++){
                                *(outputs + vl * kScalarThreadItemsK) += __half2float(lhs_value[vl] * rhs_value);
                            }
                        }
                        matrix += BlockWidth;
                    }

                    row_offsets ++;
                }

                residue -= kResidueInnerLimit_;
            }
            asm("");
        }
    };

    template <typename LoadType, int Tile_K, int Tile_N, int BlockWidth>
    struct wmmaDenseTile {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(char);

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int lane_id_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offset in shared memory
        const int* row_offsets_base_;
        // The register file fragment to load the dense values into.
        LoadType *dense_tile_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaDenseTile(
            int rhs_columns, int offset, 
            int lane_id, 
            const int* __restrict__ matrix, 
            const int *row_offsets,
            int * dense_tile):
            lane_id_(lane_id),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
            row_offsets_base_(row_offsets),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)){}
        
        // Load a pair of odd and even row groups
        __device__ __forceinline__ void LoadRow(int row_group_idx){
            const int *row_offsets = row_offsets_base_ + lane_id_/16 + row_group_idx * 16;
	    const int bank_id = lane_id_%16;
	    for(int i=0; i<8; i++){
		const int pad_offset = i/2;
                *(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
	    }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
            //if (residue == 0) return;
	    const int step = residue/2;
	    const int res_residue = residue % 2;
            const int *row_offsets = row_offsets_base_ + lane_id_/16 + row_group_idx * 16;
	    const int bank_id = lane_id_%16;
            int pad_offset = 0;
            int i = 0;
	    for(; i<step; i++){
                pad_offset = i/2;
                *(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
	    }
            pad_offset = i/2;
	    if (res_residue == 1 && lane_id_<16)
                *(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
        }
    };

    ////baseline with bank conflict
    //template <typename LoadType, int Tile_K, int Tile_N, int BlockWidth>
    //struct wmmaDenseTile_4b {
    //    //
    //    // Static members
    //    //

    //    // The number of values that will be loaded per-thread, per-load
    //    static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(char);

    //    //
    //    // Member variables
    //    //

    //    // The number of columns in the rhs matrix
    //    const int lane_id_;
    //    // The dense matrix pointer in global memory
    //    const LoadType *matrix_base_;
    //    // The loaded dense matrix row offset in shared memory
    //    const int* row_offsets_base_;
    //    // The register file fragment to load the dense values into.
    //    LoadType *dense_tile_;

    //    // Constructor. Set the initial pointer offsets
    //    __device__ __forceinline__ wmmaDenseTile_4b(
    //        int rhs_columns, int offset, 
    //        int lane_id, 
    //        const int* __restrict__ matrix, 
    //        const int *row_offsets,
    //        int * dense_tile):
    //        lane_id_(lane_id),
    //        matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
    //        row_offsets_base_(row_offsets),
    //        dense_tile_(reinterpret_cast<LoadType *>(dense_tile)){}
    //    
    //    // Load a pair of odd and even row groups
    //    __device__ __forceinline__ void LoadRow(int row_group_idx){
    //        const int *row_offsets = row_offsets_base_ + lane_id_/8 + row_group_idx * 32;
    //        const int bank_id = lane_id_%8;
    //        for(int i=0; i<8; i++){
    //            *(dense_tile_ + row_group_idx * 64 * 4 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //        }
    //    }

    //    // Load the residual and compute the matrix product
    //    __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
    //        const int step = (residue/8)*2;
    //        const int res_residue = residue % 8;
    //        const int *row_offsets = row_offsets_base_ + lane_id_/8 + row_group_idx * 32;
    //        const int bank_id = lane_id_%8;

    //        int i = 0;
    //        for(; i<step; i++){
    //            *(dense_tile_ + row_group_idx * 64 * 4 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //        }

    //        if(res_residue > 0){
    //            if (*(row_offsets + i*4) >= 0)
    //                *(dense_tile_ + row_group_idx * 64 * 4 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //            i++;
    //            if (*(row_offsets + i*4) >= 0)
    //                *(dense_tile_ + row_group_idx * 64 * 4 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //        }
    //    }
    //};
    
    //Tile_N = 128 threads_per_block = 128
    template <typename LoadType, int Tile_K, int Tile_N>
    struct wmmaDenseTile_8b{

        const int rhs_cols_;
        const int lane_id_;
        const int ints_per_row_;
        const LoadType *matrix_base_;
        const int *row_offsets_base_;
        LoadType *dense_tile_;
        int *rhs_prefetch_;

        __device__ __forceinline__ wmmaDenseTile_8b(
	    int rhs_cols,
            int offset, 
            int lane_id, 
            const int* __restrict__ matrix, 
            const int *row_offsets,
            int * dense_tile,
            int * rhs_prefetch):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            ints_per_row_(Tile_N/8),
            //ints_per_row_(Tile_N/4),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
            row_offsets_base_(row_offsets),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
            for(int i=0; i<4; i++){
                //const int pad_offset = i;
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288) = rhs_prefetch_[i];
            }
        }

        __device__ __forceinline__ void Prefetch(int step){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_ + (step % 2) * Tile_K;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            for(int i=0; i<4; i++){
                rhs_prefetch_[i] = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int residue){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            const int steps = residue / 4;
            const int res_residue = residue % 4;

	    int i = 0;
            for(; i<steps; i++){
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }

            if(res_residue > 0){
                if(*(row_offsets + i*4) >= 0)
                    *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
            //if(residue >= Tile_K){
            //    for(int i=0; i<4; i++){
            //        *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            //    }
	    //}else{
            //    const int steps = residue / 4;
            //    const int res_residue = residue % 4;
	    //    int i = 0;
            //    for(; i<steps; i++){
            //        *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            //    }

            //    if(res_residue > 0){
            //        if (*(row_offsets + i*4) >= 0)
            //            *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
	    //    }
	    //}
        }
    };

    //Tile_N = 64 threads_per_block = 128
    template <typename LoadType, int Tile_K, int Tile_N>
    struct wmmaDenseTile_16b{

        const int rhs_cols_;
        const int lane_id_;
        const int ints_per_row_;
        const LoadType *matrix_base_;
        const int *row_offsets_base_;
        LoadType *dense_tile_;
        int *rhs_prefetch_;

        __device__ __forceinline__ wmmaDenseTile_16b(
	    int rhs_cols,
            int offset, 
            int lane_id, 
            const int* __restrict__ matrix, 
            const int *row_offsets,
            int * dense_tile,
            int * rhs_prefetch):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            ints_per_row_(Tile_N/4), // for 2 warps
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
            row_offsets_base_(row_offsets),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
            for(int i=0; i<4; i++){
                //const int pad_offset = i;
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288) = rhs_prefetch_[i];
            }
        }

        __device__ __forceinline__ void Prefetch(int step){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_ + (step % 2) * Tile_K;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            for(int i=0; i<4; i++){
                rhs_prefetch_[i] = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int residue){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            const int steps = residue / 4;
            const int res_residue = residue % 4;

	    int i = 0;
            for(; i<steps; i++){
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }

            if(res_residue > 0){
                if(*(row_offsets + i*4) >= 0)
                    *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
        }
    };

    ////Tile_N = 64 threads_per_block = 64
    //template <typename LoadType, int Tile_K, int Tile_N>
    //struct wmmaDenseTile_8b{

    //    const int rhs_cols_;
    //    const int lane_id_;
    //    const int ints_per_row_;
    //    const LoadType *matrix_base_;
    //    const int *row_offsets_base_;
    //    LoadType *dense_tile_;
    //    int *rhs_prefetch_;

    //    __device__ __forceinline__ wmmaDenseTile_8b(
    //        int rhs_cols,
    //        int offset, 
    //        int lane_id, 
    //        const int* __restrict__ matrix, 
    //        const int *row_offsets,
    //        int * dense_tile,
    //        int * rhs_prefetch):
    //        rhs_cols_(rhs_cols),
    //        lane_id_(lane_id),
    //        ints_per_row_(Tile_N/4),
    //        matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
    //        row_offsets_base_(row_offsets),
    //        dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
    //        rhs_prefetch_(rhs_prefetch){}
    //    

    //    __device__ __forceinline__ void LoadRowfromRegister(int step){
    //        for(int i=0; i<4; i++){
    //            //const int pad_offset = i;
    //            *(dense_tile_ + i*72 + lane_id_) = rhs_prefetch_[i];
    //        }
    //    }

    //    __device__ __forceinline__ void Prefetch(int step){
    //        const int *row_offsets = row_offsets_base_ + lane_id_/ints_per_row_ + (step % 2) * Tile_K;
    //        const int bank_id = lane_id_%ints_per_row_;
    //        for(int i=0; i<4; i++){
    //            rhs_prefetch_[i] = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
    //        }
    //    }

    //    // Load the residual and compute the matrix product
    //    __device__ __forceinline__ void ResidueLoad(int residue){
    //        const int *row_offsets = row_offsets_base_ + lane_id_/ints_per_row_;
    //        const int bank_id = lane_id_%ints_per_row_;
    //        const int steps = residue / 4;
    //        const int res_residue = residue % 4;

    //        int i = 0;
    //        for(; i<steps; i++){
    //            *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
    //        }

    //        if(res_residue > 0){
    //            if (*(row_offsets + i*4) >= 0)
    //                *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
    //        }
    //        //if(residue >= Tile_K){
    //        //    for(int i=0; i<4; i++){
    //        //        *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
    //        //    }
    //        //}else{
    //        //    const int steps = residue / 4;
    //        //    const int res_residue = residue % 4;
    //        //    int i = 0;
    //        //    for(; i<steps; i++){
    //        //        *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
    //        //    }

    //        //    if(res_residue > 0){
    //        //        if (*(row_offsets + i*4) >= 0)
    //        //            *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
    //        //    }
    //        //}
    //    }
    //};

    //larger Tile_N 128
    template <typename LoadType, int Tile_K, int Tile_N>
    struct wmmaDenseTile_4b{
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        //static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(char);
        //static constexpr int kTotalStep = Tile_K / 16 - 1;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int lane_id_;
        const int rhs_cols_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offset in shared memory
        const int *row_offsets_base_;
        // The register file fragment to load the dense values into.
        LoadType *dense_tile_;
        int *rhs_prefetch_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaDenseTile_4b(
	    int rhs_cols,
            int offset, 
            int lane_id, 
            const int* __restrict__ matrix, 
            const int *row_offsets,
            int * dense_tile,
            int * rhs_prefetch):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
            row_offsets_base_(row_offsets),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
            for(int i=0; i<8; i++){
                const int pad_offset = i/2;
                *(dense_tile_ + pad_offset*8 + lane_id_ + i*64) = rhs_prefetch_[i];
            }
        }

        __device__ __forceinline__ void Prefetch(int step){
            const int *row_offsets = row_offsets_base_ + lane_id_/16 + (step % 2) * Tile_K;
            const int bank_id = lane_id_%16;
            for(int i=0; i<8; i++){
                rhs_prefetch_[i] = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int residue){
            const int steps = (residue/8)*2;
            const int res_residue = residue % 8;
            const int *row_offsets = row_offsets_base_ + lane_id_/16;
            const int bank_id = lane_id_%16;

            int pad_offset = 0;
            int i = 0;
            for(; i<steps; i++){
                pad_offset = i/2;
                *(dense_tile_ + pad_offset*8 + lane_id_ + i*64) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            }

            if(res_residue > 0){
                pad_offset = i/2;
                if (*(row_offsets + i*4) >= 0)
                    *(dense_tile_ + pad_offset*8 + lane_id_ + i*64) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
                i++;
                pad_offset = i/2;
                if (*(row_offsets + i*4) >= 0)
                    *(dense_tile_ + pad_offset*8 + lane_id_ + i*64) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            }
        }
    };

    //template <typename LoadType, int Tile_K, int Tile_N, int BlockWidth>
    //struct wmmaDenseTile_4b {
    //    //
    //    // Static members
    //    //

    //    // The number of values that will be loaded per-thread, per-load
    //    static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(char);
    //    static constexpr int kTotalStep = Tile_K / 16 - 1;

    //    //
    //    // Member variables
    //    //

    //    // The number of columns in the rhs matrix
    //    const int lane_id_;
    //    // The dense matrix pointer in global memory
    //    const LoadType *matrix_base_;
    //    // The loaded dense matrix row offset in shared memory
    //    const int* row_offsets_base_;
    //    // The register file fragment to load the dense values into.
    //    LoadType *dense_tile_;

    //    // Constructor. Set the initial pointer offsets
    //    __device__ __forceinline__ wmmaDenseTile_4b(
    //        int rhs_columns, int offset, 
    //        int lane_id, 
    //        const int* __restrict__ matrix, 
    //        const int *row_offsets,
    //        int * dense_tile):
    //        lane_id_(lane_id),
    //        matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
    //        row_offsets_base_(row_offsets),
    //        dense_tile_(reinterpret_cast<LoadType *>(dense_tile)){}
    //    
    //    // Load a pair of odd and even row groups
    //    __device__ __forceinline__ void LoadRow(int row_group_idx){
    //        const int *row_offsets = row_offsets_base_ + lane_id_/8 + row_group_idx * 32;
    //        const int bank_id = lane_id_%8;
    //        for(int i=0; i<8; i++){
    //    	const int pad_offset = i/2;
    //            //*(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //            *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //        }
    //    }

    //    // Load the residual and compute the matrix product
    //    __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
    //        const int step = (residue/8)*2;
    //        const int res_residue = residue % 8;
    //        const int *row_offsets = row_offsets_base_ + lane_id_/8 + row_group_idx * 32;
    //        const int bank_id = lane_id_%8;

    //        int pad_offset = 0;
    //        int i = 0;
    //        for(; i<step; i++){
    //            pad_offset = i/2;
    //            //*(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //            *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //        }

    //        if(res_residue > 0){
    //            pad_offset = i/2;
    //            if (*(row_offsets + i*4) >= 0)
    //                //*(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //                *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //            i++;
    //            pad_offset = i/2;
    //            if (*(row_offsets + i*4) >= 0)
    //                //*(dense_tile_ + row_group_idx * 72 * 4 + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //                *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*4) + bank_id);
    //        }
    //    }
    //};
}

#endif
