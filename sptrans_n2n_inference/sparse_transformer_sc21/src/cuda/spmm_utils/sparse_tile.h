#ifndef SPMM_SPARSE_TILE_H
#define SPMM_SPARSE_TILE_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace spmm{
    __device__ __forceinline__ void Mul(int x1, int2 x2, int2 *out) {
        out[0].x = x1 * x2.x;
        out[0].y = x1 * x2.y;
    }

    __device__ __forceinline__ void Mul(int x1, int4 x2, int4 *out) {
        out[0].x = x1 * x2.x;
        out[0].y = x1 * x2.y;
        out[0].z = x1 * x2.z;
        out[0].w = x1 * x2.w;
    }

    __device__ __forceinline__ void Mul(int x1, int x2, int *out) {
        out[0] = x1 * x2;
    }

    template <typename LoadType, typename IndexType, typename VecType, int VecLength, int Tile_N, int BlockWidth>
    struct SparseTile {
        // 
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kIndicesPerLoad_ = sizeof(IndexType) / sizeof(int);
        // When the LoadType is half8, there is no int8 type. So we need to load the index in two steps
        static constexpr int kRatio = kValuesPerLoad_ / kIndicesPerLoad_ / VecLength;
        // The number of data items in the n-dimension that each thread owns
        static constexpr int kThreadItemsN_ = Tile_N * VecLength / BlockWidth / kValuesPerLoad_;
        static constexpr int kScalarThreadItemsN_ = Tile_N / BlockWidth;

        // 
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const LoadType *values_;
        // The sparse matrix column indices for each value. 
        const IndexType *column_idxs_;
        // shared memory tile for sparse matrix values
        LoadType *values_tile_base_;
        // shared memory tile for sparse matrix indices
        IndexType *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets.
        __device__ __forceinline__ SparseTile(
            int rhs_columns, int row_offset_vec, int thread_idx_x,
            const half* __restrict__ values,
            const int* __restrict__ column_idxs,
            half *values_tile, int*column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const LoadType *>(values + row_offset_vec * VecLength) + thread_idx_x),
            column_idxs_(reinterpret_cast<const IndexType *>(column_idxs + row_offset_vec) + thread_idx_x * kRatio),
            values_tile_base_(reinterpret_cast<LoadType *>(values_tile) + thread_idx_x),
            column_idxs_tile_base_(reinterpret_cast<IndexType *>(column_idxs_tile) + thread_idx_x * kRatio){}

        // Load
        __device__ __forceinline__ void Load(){
            LoadType *values_tile = values_tile_base_;
            IndexType *column_idxs_tile = column_idxs_tile_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                *(values_tile) = __ldg(values_);
                #pragma unroll
                for (int r = 0; r < kRatio; r++){
                    Mul(rhs_columns_, __ldg(column_idxs_ + r), column_idxs_tile + r);
                }
                values_ += BlockWidth;
                column_idxs_ += kRatio * BlockWidth;
                values_tile += BlockWidth;
                column_idxs_tile += kRatio * BlockWidth;
            }
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            LoadType *values_tile = values_tile_base_;
            IndexType *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[kValuesPerLoad_] = {};
            const int kZerosIndices[kIndicesPerLoad_] = {};

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                *(values_tile) = reinterpret_cast<const LoadType*>(kZeroValues)[0];
                #pragma unroll
                for (int r = 0; r < kRatio; r++){
                    *(column_idxs_tile + r) = reinterpret_cast<const IndexType*>(kZerosIndices)[0];
                }
                values_tile += BlockWidth;
                column_idxs_tile += kRatio * BlockWidth;
            }
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            constexpr int kResidueUpdateStride = -1 * (kValuesPerLoad_ / VecLength - 1);
            constexpr int kIndexResidueUpdateStride = -1 * (kIndicesPerLoad_ * kRatio - 1);
            const int kInitOffsetAdjust = static_cast<int>(threadIdx.x) * kResidueUpdateStride;
            const int kIndexInitOffsetAdjust = static_cast<int>(threadIdx.x) * kIndexResidueUpdateStride;

            const VecType* values = reinterpret_cast<const VecType* >(values_) + kInitOffsetAdjust;
            const int* column_idxs = reinterpret_cast<const int*>(column_idxs_) + kIndexInitOffsetAdjust;

            VecType* values_tile = reinterpret_cast<VecType *>(values_tile_base_) + kInitOffsetAdjust;
            int* column_idxs_tile = reinterpret_cast<int *>(column_idxs_tile_base_) + kIndexInitOffsetAdjust;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kScalarThreadItemsN_; n_item_idx ++){
                if (residue <= threadIdx.x) return;
                *(values_tile) = __ldg(values);
                *(column_idxs_tile) = __ldg(column_idxs) * rhs_columns_;

                values += BlockWidth;
                column_idxs += BlockWidth;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
                residue -= BlockWidth;
            }
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }
    };

    template <typename LoadType, typename VecType, int VecLength, int Tile_N, int BlockWidth>
    struct SparseTile1D {
        //
        // Static members
        //

        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const VecType* values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        VecType* values_tile_base_;
        // shared memory tile for sparse marix values
        int *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ SparseTile1D(
            int rhs_columns, int row_offset_vec, int thread_idx_x,
            const half* __restrict__ values,
            const int* __restrict__ column_idxs,
            VecType *values_tile, int * column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const VecType *>(values) + row_offset_vec + thread_idx_x),
            column_idxs_(reinterpret_cast<const int *>(column_idxs) + row_offset_vec + thread_idx_x),
            values_tile_base_(reinterpret_cast<VecType *>(values_tile) + thread_idx_x),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + thread_idx_x){}
        
        // Load
        __device__ __forceinline__ void Load(){
            VecType *values_tile = values_tile_base_;
            int* column_idxs_tile = column_idxs_tile_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                *(values_tile) = __ldg(values_);
                *(column_idxs_tile) = rhs_columns_ * __ldg(column_idxs_);
                values_ += BlockWidth;
                column_idxs_ += BlockWidth;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
            }
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            VecType *values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[VecLength] = {};
            
            #pragma unrill
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                *(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
                *(column_idxs_tile) = 0;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
            }
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            VecType* values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                if (residue <= threadIdx.x) return;
                *(values_tile) = __ldg(values_);
                *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;

                values_ += BlockWidth;
                column_idxs_ += BlockWidth;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
                residue -= BlockWidth;
            }
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }

    };

    template <typename LoadType, typename VecType, int VecLength, int Tile_N, int BlockWidth>
    struct wmmaSparseTile {
        //
        // Static members
        //

        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const VecType* values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        VecType* values_tile_base_;
        // shared memory tile for sparse marix values
        int *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaSparseTile(
            int rhs_columns, int row_offset_vec, int thread_idx_x,
            const half* __restrict__ values,
            const int* __restrict__ column_idxs,
            VecType *values_tile, int * column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const VecType *>(values) + row_offset_vec + thread_idx_x),
            column_idxs_(reinterpret_cast<const int *>(column_idxs) + row_offset_vec + thread_idx_x),
            values_tile_base_(reinterpret_cast<VecType *>(values_tile) + thread_idx_x),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + thread_idx_x){}
        
        // Load
        __device__ __forceinline__ void Load(){
            VecType *values_tile = values_tile_base_;
            int* column_idxs_tile = column_idxs_tile_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                *(values_tile) = __ldg(values_);
                *(column_idxs_tile) = rhs_columns_ * __ldg(column_idxs_);
                values_ += BlockWidth;
                column_idxs_ += BlockWidth;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
            }
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            VecType *values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[VecLength] = {};
            
            #pragma unrill
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                *(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
                *(column_idxs_tile) = 0;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
            }
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            VecType* values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
                if (residue <= threadIdx.x) return;
                *(values_tile) = __ldg(values_);
                *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;

                values_ += BlockWidth;
                column_idxs_ += BlockWidth;
                values_tile += BlockWidth;
                column_idxs_tile += BlockWidth;
                residue -= BlockWidth;
            }
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }

    };
}
#endif