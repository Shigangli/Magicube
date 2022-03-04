#ifndef SPMM_DENSE_TILE_H
#define SPMM_DENSE_TILE_H

#include <cuda_fp16.h>
#include <stdlib.h>
#include <stdio.h>

namespace spmm {
    template <typename LoadType, typename VecType, int Tile_N, int Tile_K, int BlockWidth, int VecLength>
    struct DenseTile {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsK = Tile_N / BlockWidth / kValuesPerLoad_;
        static constexpr int kScalarThreadItemsK = Tile_N / BlockWidth;
        static constexpr int kResidueUnroll = 4;
        static constexpr int kResidueOuterLimit_ = Tile_N / kResidueUnroll;
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
            for (int n_item_idx = 0; n_item_idx < Tile_N; n_item_idx ++){
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

    template <typename LoadType, typename VecType, int Tile_N, int Tile_K, int BlockWidth, int VecLength>
    struct DenseTile1D {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsK = Tile_N / BlockWidth / kValuesPerLoad_;
        static constexpr int kScalarThreadItemsK = Tile_N / BlockWidth;
        static constexpr int kResidueUnroll = 4;
        static constexpr int kResidueOuterLimit_ = Tile_N / kResidueUnroll;
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
            for (int n_item_idx = 0; n_item_idx < Tile_N; n_item_idx ++){
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

    template <typename LoadType, int Tile_N, int Tile_K, int BlockWidth>
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
    //template <typename LoadType, int Tile_N, int Tile_K, int BlockWidth>
    //struct wmmaDenseTile_4bit {
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
    //    __device__ __forceinline__ wmmaDenseTile_4bit(
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

    //larger Tile_K 128 sm opt
    template <typename LoadType>
    struct wmmaDenseTile_4bit_sm_opt {
        //
        // Static members
        //

        //
        // Member variables
        //
        // The number of columns in the rhs matrix
        const int rhs_columns_;
        const int lane_num_;
        const int lane_id_;
        const int local_thread_id_;
        const int row_id_;
        const int row_num_;
        const int col_index_offset_begin_;
        const int sm_rows_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offset in shared memory
        const int* col_indices_;
        const int* compact_block_offset_;
        // The register file fragment to load the dense values into.
        LoadType *dense_tile_;
        int *offsets_tile_;
        int col_idx_offset_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaDenseTile_4bit_sm_opt(
            int rhs_columns, int offset, 
            int local_thread_id, 
            int local_thread_num, 
            int lane_num, 
            int col_index_offset_begin,
            int sm_rows,
            const int * __restrict__ rhs_matrix, 
            const int *col_indices,
            const int *compact_block_offset,
            int * dense_tile,
            int * offsets_tile):
	    rhs_columns_(rhs_columns),
	    lane_num_(lane_num),
            local_thread_id_(local_thread_id),
	    lane_id_(local_thread_id % lane_num),
	    row_id_(local_thread_id / lane_num),
            row_num_(local_thread_num / lane_num),
            col_index_offset_begin_(col_index_offset_begin),
            sm_rows_(sm_rows),
            matrix_base_(reinterpret_cast<const LoadType *>(rhs_matrix + offset)),
            col_indices_(col_indices),
            compact_block_offset_(compact_block_offset),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
            offsets_tile_(offsets_tile){}
        
        // Load a pair of odd and even row groups
        __device__ __forceinline__ void LoadRow(int row_block_offset){
	    int col_index_offset = __ldg(compact_block_offset_ + row_block_offset);
	    int col_index_num = __ldg(compact_block_offset_ + row_block_offset + 1) - col_index_offset;
	    int residual = col_index_num % row_num_;
	    int chunk_num = col_index_num / row_num_;

	    int sm_row_offset = (col_index_offset - col_index_offset_begin_ + row_id_) % sm_rows_;
	    //int lhs_matrix_row_offset;
	    if(local_thread_id_ < col_index_num)
                *(offsets_tile_ + local_thread_id_) = rhs_columns_ * __ldg(col_indices_ + col_index_offset + local_thread_id_);
	    __syncthreads();

	    //lhs_matrix_row_offset = __ldg(col_indices_ + col_index_offset + row_id_) * rhs_columns_;
	    //lhs_matrix_row_offset = __ldg(col_indices_ + col_index_offset + row_id_) * rhs_columns_;
	    //printf("chunk_num = %d \n", chunk_num);
	    //printf("lane_num_ = %d ", lane_num_);
	    //printf("row_id_ = %d \n", row_id_);
	    //printf("row_num_ = %d \n", row_num_);
            
	    //if(*(offsets_tile_) >=0){
	        int i;
                for(i=0; i<chunk_num; i++){
	            //lhs_matrix_row_offset = __ldg(col_indices_ + col_index_offset + row_id_) * rhs_columns_;
	            //*(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + lhs_matrix_row_offset + lane_id_); 
	            //*(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + sm_row_offset * rhs_columns_ + lane_id_); 
	            //lhs_matrix_row_offset = *(offsets_tile_ + load_row_id);
	            *(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + *(offsets_tile_ + row_id_ + i*row_num_) + lane_id_); 
	            //*(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + (sm_row_offset + row_id_) * rhs_columns_ + lane_id_); 

	            //sm_row_offset = (sm_row_offset + row_num_) % sm_rows_;
	            sm_row_offset += row_num_;
	            if(sm_row_offset >= sm_rows_)
                        sm_row_offset -= sm_rows_;
                }

	        if(row_id_ < residual){
	            //lhs_matrix_row_offset = __ldg(col_indices_ + col_index_offset + row_id_) * rhs_columns_;

	            //lhs_matrix_row_offset = *(offsets_tile_ + load_row_id);
	            //*(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + lhs_matrix_row_offset + lane_id_); 
	            //*(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + *(offsets_tile_ + load_row_id) + lane_id_);

	            *(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + *(offsets_tile_ + row_id_ + i*row_num_) + lane_id_); 
	            //*(dense_tile_ + sm_row_offset * lane_num_ + lane_id_) = __ldg(matrix_base_ + (sm_row_offset + row_id_) * rhs_columns_ + lane_id_); 
	        }
	    //}
	    
        }
    };


    //larger Tile_K 128
    template <typename LoadType, int Tile_N, int Tile_K, int BlockWidth>
    struct wmmaDenseTile_4bit {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(char);
        static constexpr int kTotalStep = Tile_N / 16 - 1;

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
        __device__ __forceinline__ wmmaDenseTile_4bit(
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
            const int *row_offsets = row_offsets_base_ + lane_id_/16 + row_group_idx * 32;
            const int bank_id = lane_id_%16;
            for(int i=0; i<16; i++){
        	const int pad_offset = i/4;
                *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
            const int step = (residue/8)*4;
            const int res_residue = residue % 8;
            const int *row_offsets = row_offsets_base_ + lane_id_/16 + row_group_idx * 32;
            const int bank_id = lane_id_%16;

            int pad_offset = 0;
            int i = 0;
            for(; i<step; i++){
                pad_offset = i/4;
                *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
            }

            if(res_residue > 0){
                pad_offset = i/4;
                if (*(row_offsets + i*2) >= 0)
                    *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
                i++;
                pad_offset = i/4;
                if (*(row_offsets + i*2) >= 0)
                    *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
                i++;
                pad_offset = i/4;
                if (*(row_offsets + i*2) >= 0)
                    *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
                i++;
                pad_offset = i/4;
                if (*(row_offsets + i*2) >= 0)
                    *(dense_tile_ + pad_offset*8 + lane_id_ + i*32) = __ldg(matrix_base_ + *(row_offsets + i*2) + bank_id);
            }
        }
    };

    //tile_K = 64
    //template <typename LoadType, int Tile_N, int Tile_K, int BlockWidth>
    //struct wmmaDenseTile_4bit {
    //    //
    //    // Static members
    //    //

    //    // The number of values that will be loaded per-thread, per-load
    //    static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(char);
    //    static constexpr int kTotalStep = Tile_N / 16 - 1;

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
    //    __device__ __forceinline__ wmmaDenseTile_4bit(
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
