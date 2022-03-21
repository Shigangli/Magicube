#ifndef SPMM_OUTPUT_Tile_H
#define SPMM_OUTPUT_Tile_H

#include <cuda_fp16.h>

namespace spmm{
    
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

            #pragma unroll
            for(int i=0; i<8; i++){
                deq_results[i] = __float2half((float)(output_fragment_[i]) / scale_);
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
            if(lane_id_ % 32 < valid_tsize_){
                for(int i = 0; i < 4; i++)
                    final_output_fragment_[i] += (final_output_fragment_[i+4] * 256);
            }

            half deq_results[4] = {};
            #pragma unroll
            for(int i=0; i<4; i++){
                deq_results[i] = __float2half((float)(final_output_fragment_[i]) / scale_);
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
        int2* output_matrix_;
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
            output_matrix_ = reinterpret_cast<int2 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){

            half deq_results[16] = {};
            #pragma unroll
            for(int i=0; i<16; i++){
                deq_results[i] = __float2half((float)(output_fragment_[i]) / scale_);
            }

            int output_off = (lane_id_ % 4) * 4 + (lane_id_ / 32) * 16;
            if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int2 *>(deq_results) + 0);
                *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int2 *>(deq_results) + 1);
                *(output_matrix_ + output_off + 2) = *(reinterpret_cast<int2 *>(deq_results) + 2);
                *(output_matrix_ + output_off + 3) = *(reinterpret_cast<int2 *>(deq_results) + 3);
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

            for(int i = 0; i < 16; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 16);

            half deq_results[16] = {};
            #pragma unroll
            for(int i=0; i<16; i++){
                deq_results[i] = __float2half((float)(output_fragment_0_[i]) / scale_);
            }

            int output_off = (lane_id_ % 4) * 4 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int2 *>(deq_results) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int2 *>(deq_results) + 1);
            *(output_matrix_ + output_off + 2) = *(reinterpret_cast<int2 *>(deq_results) + 2);
            *(output_matrix_ + output_off + 3) = *(reinterpret_cast<int2 *>(deq_results) + 3);
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
        int2* output_matrix_;
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
            output_matrix_ = reinterpret_cast<int2 *>(output_matrix + output_offset);
            lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
            scale_ = scale;
        }

        // Store
        __device__ __forceinline__ void Store(){
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 0], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 1], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 2], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 3], half_valid_tsize_, 32);

            int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);

            half deq_results[8] = {};

            if(lane_id_ % 32 < valid_tsize_){
                for(int i = 0; i < 8; i++){
                    final_output_fragment_[i] += (final_output_fragment_[i+8] * 16);
                    deq_results[i] = __float2half((float)(final_output_fragment_[i]) / scale_);
                }
            }

            int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / half_valid_tsize_) * 2 + (lane_id_ / 32) * 16;
            if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int2 *>(deq_results) + 0);
                *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int2 *>(deq_results) + 1);
            }
        }
    };

}

#endif
