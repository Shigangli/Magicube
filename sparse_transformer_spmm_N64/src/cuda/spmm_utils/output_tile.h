#ifndef SPMM_OUTPUT_Tile_H
#define SPMM_OUTPUT_Tile_H

#include <cuda_fp16.h>

    template <typename LoadType, typename OutType, int Tile_K, int BlockWidth, int VecLength>
    struct OutputTile{

        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(LoadType) / sizeof(OutType);
        static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsK_ = Tile_K / BlockWidth / kValuesPerStore_;
        static constexpr int kScaler_ = sizeof(OutType)/sizeof(half);

        //
        // Member variables
        //

        // The register file fragment with the results to store
        const LoadType* output_fragment_;
        // The output matrix pointer in global memory
        LoadType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ OutputTile(
            int m_index_vec, int column_offset,
            int cols, int thread_idx_x,
            const float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<const LoadType *>(output_fragment);
            const int output_offset = m_index_vec * VecLength * cols + column_offset;
            output_matrix_ = reinterpret_cast<LoadType *>(output_matrix + output_offset) + thread_idx_x * kScaler_;
            rhs_columns_ = cols / kValuesPerStore_ * kScaler_;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if (kValuesPerLoad == kValuesPerStore_){
                #pragma unroll
                for (int v = 0; v < VecLength; v++){
                    const LoadType * output_fragment_t = output_fragment_ + v * 2 * kThreadItemsK_;
                    LoadType * output_matrix_t = output_matrix_ + v * rhs_columns_;
                    #pragma unroll
                    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                        float values [kValuesPerStore_];
                        LoadType * values_loadType = reinterpret_cast<LoadType *>(values);
                        OutType *values_outType = reinterpret_cast<OutType *>(values);
                        *(values_loadType) = *(output_fragment_t);
                        *(values_loadType + 1) = *(output_fragment_t + 1);
                        #pragma unroll 
                        for (int dv = 0; dv < kValuesPerStore_; dv ++){
                            values_outType[dv] = (OutType)values[dv];
                        }
                        *(output_matrix_t) = *(values_loadType);
                        output_fragment_t += 2;
                        output_matrix_t +=BlockWidth;
                    }
                }
            }
            else{
                #pragma unroll
                for (int v = 0; v < VecLength; v++){
                    const LoadType * output_fragment_t = output_fragment_ + v * 2 * kThreadItemsK_;
                    LoadType * output_matrix_t = output_matrix_ + v * rhs_columns_;
                    #pragma unroll
                    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                        *(output_matrix_t) = *(output_fragment_t);
                        *(output_matrix_t + 1) = *(output_fragment_t + 1);
                        output_matrix_t +=BlockWidth * 2;
                        output_fragment_t += 2;
                    }
                }
            }
        }
    };
    
    // 4 warps Tile_N = 128 8-bit v=2 4 8
    struct wmmaOutputTile_8b{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        // The register file fragment with the results to store
        int* output_fragment_;
        int2* output_matrix_;
        float scale_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            half* output_matrix,
            float scale)
        {
            output_fragment_ = output_fragment;
            valid_tsize_ = 4 * vec_length; // =32/(8/vec_length);
            const int output_offset = (m_index_vec * vec_length + (lane_id % 32) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int2 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){
            
            half deq_results[8] = {};

            if(lane_id_ % 32 < valid_tsize_){
                for(int i=0; i<8; i++){
                    deq_results[i] = __float2half((float)(output_fragment_[i]) / scale_);
                }
            }

            int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
            if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int2 *>(deq_results) + 0);
                *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int2 *>(deq_results) + 1);
            }
        }
    };

    // 4 warps Tile_N = 128 16-bit 8-bit v=2 4
    struct wmmaOutputTile_16b8b{
        //
        // Member variables
        //

        int lane_id_;
        int valid_tsize_;
        int half_valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int2* output_matrix_;
        float scale_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b8b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            half* output_matrix,
            float scale)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
            valid_tsize_ = 8 * vec_length; // =32/(8/vec_length)*2;
            half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset; //32/(8/vec_length)
            output_matrix_ = reinterpret_cast<int2 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){

            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 0], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 1], half_valid_tsize_, 32);

            int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);
            half deq_results[4] = {};

            if(lane_id_ % 32 < valid_tsize_){
                for(int i = 0; i < 4; i++){
                    final_output_fragment_[i] += (final_output_fragment_[i+4] * 256);
                    deq_results[i] = __float2half((float)(final_output_fragment_[i]) / scale_);
                }
            }

            int output_off = (intra_warp_tid_ % 4) * 2 + (intra_warp_tid_ / half_valid_tsize_) + (lane_id_ / 32) * 8;
            if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off) = *(reinterpret_cast<int2 *>(deq_results));
            }
        }
    };


    // 4 warps Tile_N = 128 16-bit 8-bit v=8
    struct wmmaOutputTile_16b8b8v{
        //
        // Member variables
        //

        int lane_id_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int2* output_matrix_;
        float scale_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b8b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            half* output_matrix,
            float scale)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
            const int output_offset = (m_index_vec * vec_length + (lane_id % 32) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int2 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){
            for(int i = 0; i < 8; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 256);

            half deq_results[8] = {};
            #pragma unroll
            for(int i=0; i<8; i++){
                deq_results[i] = __float2half((float)(output_fragment_0_[i]) / scale_);
            }

            int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int2 *>(deq_results) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int2 *>(deq_results) + 1);
        }
    };

    

    // Tile_N = 128 4-bit 2 warps
    struct wmmaOutputTile_4b{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        // The register file fragment with the results to store
        int* output_fragment_;
        int4* output_matrix_;
        float scale_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_4b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            half* output_matrix,
            float scale)
        {
            output_fragment_ = output_fragment;
            valid_tsize_ = 4 * vec_length; // =32/(8/vec_length);
            const int output_offset = (m_index_vec * vec_length + (lane_id % 32) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){

            half deq_results[8] = {};
            
            if(lane_id_ % 32 < valid_tsize_){
                for(int i=0; i<8; i++){
                    deq_results[i] = __float2half((float)(output_fragment_[i]) / scale_);
                }
            }

            int output_off = lane_id_ % 4 + (lane_id_ / 32) * 4;
            if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off) = *(reinterpret_cast<int4 *>(deq_results));
            }
        }
    };

    struct wmmaOutputTile_8b4b8v{
        //
        // Member variables
        //
        int lane_id_;
        int half_valid_tsize_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int2* output_matrix_;
        float scale_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b4b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            half* output_matrix,
            float scale)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
            half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset; //32/(8/vec_length)
            output_matrix_ = reinterpret_cast<int2 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){

            for(int i = 0; i < 8; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 16);

            half deq_results[8] = {};
            #pragma unroll
            for(int i=0; i<8; i++){
                deq_results[i] = __float2half((float)(output_fragment_0_[i]) / scale_);
            }

            int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int2 *>(deq_results) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int2 *>(deq_results) + 1);
        }
    };

    struct wmmaOutputTile_8b4b{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        int half_valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int* output_matrix_;
        float scale_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b4b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            half* output_matrix,
            float scale)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
            valid_tsize_ = 8 * vec_length; // =32/(8/vec_length)*2;
            half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 0], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*2 + 1], half_valid_tsize_, 32);

            int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);

            half deq_results[4] = {};

            if(lane_id_ % 32 < valid_tsize_){
                for(int i = 0; i < 4; i++){
                    final_output_fragment_[i] += (final_output_fragment_[i+4] * 16);
                    deq_results[i] = __float2half((float)(final_output_fragment_[i]) / scale_);
                }
            }

            int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / half_valid_tsize_) * 2 + (lane_id_ / 32) * 16;
            if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int *>(deq_results) + 0);
                *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int *>(deq_results) + 1);
            }
        }
    };

#endif
