#ifndef SPMM_OUTPUT_Tile_H
#define SPMM_OUTPUT_Tile_H

namespace spmm{
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
            int row_offset_vec, int column_offset,
            int cols, int thread_idx_x,
            const float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<const LoadType *>(output_fragment);
            const int output_offset = row_offset_vec * VecLength * cols + column_offset;
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

    template<typename OutType, typename StoreType>
    struct wmmaOutputTile8{
        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float2* output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile8(
            int lane_id, int thread_group, 
            int row_offset_vec, int column_offset,
            int cols, 
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);
            rhs_columns_ = cols / kValuesPerStore_;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
            int src_line = (lane_id_ + 2) % 4 + thread_group_ * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++){
                __align__(8) float temp[2];
                float2* temp_float2 = reinterpret_cast<float2 *>(temp);

                if (lane_id_ < 2) *(temp_float2) = output_fragment_[i * 2 + 1];
                else *(temp_float2) = output_fragment_[i * 2];
                temp[0] = __shfl_sync(0xffffffff, temp[0], src_line, 32);
                temp[1] = __shfl_sync(0xffffffff, temp[1], src_line, 32);
                if (lane_id_ < 2) output_fragment_[i * 2 + 1] = *(temp_float2);
                else output_fragment_[i * 2] = *(temp_float2);
            }

            if (kTypeConvert != 1){
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = __float2half(output_fragment_float[i]);
                }
            }
            

            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            *(output_matrix_) = *(output_fragment_storetype);
            *(output_matrix_ + 1) = *(output_fragment_storetype + 2);
            *(output_matrix_ + 8) = *(output_fragment_storetype + 1);
            *(output_matrix_ + 9) = *(output_fragment_storetype + 3);
            
        }
    };

    template<typename OutType>
    struct wmmaOutputTile4{
        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float* output_fragment_;
        OutType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile4(
            int lane_id, int thread_group,
            int row_offset_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float *>(output_fragment);
            const int output_offset = (row_offset_vec * 4) * cols + column_offset + thread_group * 8 + lane_id;
            output_matrix_ = reinterpret_cast<OutType *>(output_matrix + output_offset);
            rhs_columns_ = cols;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
            float2* output_fragment_float2 = reinterpret_cast<float2 *>(output_fragment_);
            int src_line = (lane_id_ + 2) % 4 + thread_group_ * 4;
            #pragma unroll
            for (int i = 0; i < 2; i++){
                __align__(8) float temp[2];
                float2* temp_float2 = reinterpret_cast<float2 *>(temp);

                if (lane_id_ < 2) *(temp_float2) = output_fragment_float2[i * 2 + 1];
                else *(temp_float2) = output_fragment_float2[i * 2];
                temp[0] = __shfl_sync(0xffffffff, temp[0], src_line, 32);
                temp[1] = __shfl_sync(0xffffffff, temp[1], src_line, 32);
                if (lane_id_ < 2) output_fragment_float2[i * 2 + 1] = *(temp_float2);
                else output_fragment_float2[i * 2] = *(temp_float2);
            }

            #pragma unroll
            for (int i = 0; i < 4; i++){
                *(output_matrix_ + i * rhs_columns_) = __float2half(*(output_fragment_ + i));
                *(output_matrix_ + 4 + i * rhs_columns_) = __float2half(*(output_fragment_ + 4 + i));
            }
        }
    };

    template<typename OutType>
    struct wmmaOutputTile2{
        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float* output_fragment_;
        OutType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile2(
            int lane_id, int thread_group,
            int row_offset_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float *>(output_fragment);
            const int output_offset = (row_offset_vec * 2) * cols + column_offset + thread_group * 8 + lane_id;
            output_matrix_ = reinterpret_cast<OutType *>(output_matrix + output_offset);
            rhs_columns_ = cols;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
            int src_line = (lane_id_ + 2) % 4 + thread_group_ * 4;
            
            #pragma unroll
            for (int i = 0; i < 2; i++){
                float temp;
                if (lane_id_ < 2) temp = output_fragment_[i * 2 + 1];
                else temp = output_fragment_[i * 2];
                temp = __shfl_sync(0xffffffff, temp, src_line, 32);
                if (lane_id_ < 2) output_fragment_[i * 2 + 1] = temp;
                else output_fragment_[i * 2] = temp;
            }

            #pragma unroll
            for (int i = 0; i < 2; i++){
                *(output_matrix_ + i * rhs_columns_) = __float2half(*(output_fragment_ + i));
                *(output_matrix_ + 4 + i * rhs_columns_) = __float2half(*(output_fragment_ + 2 + i));
            }
        }
    };
}
#endif