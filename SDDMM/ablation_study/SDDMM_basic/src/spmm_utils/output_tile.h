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
            int m_index_vec, int column_offset,
            int cols, 
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            const int output_offset = (m_index_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
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
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
            }
            

            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            *(output_matrix_) = *(output_fragment_storetype);
            *(output_matrix_ + 1) = *(output_fragment_storetype + 2);
            *(output_matrix_ + 8) = *(output_fragment_storetype + 1);
            *(output_matrix_ + 9) = *(output_fragment_storetype + 3);
            
        }
    };


    // 4 warps Tile_N = 128 8-bit v=2 4 8
    template<typename OutType>
    struct wmmaOutputTile_8b{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        // The register file fragment with the results to store
        int* output_fragment_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = output_fragment;
	    valid_tsize_ = 4 * vec_length; // =32/(8/vec_length);
            const int output_offset = (m_index_vec * vec_length + (lane_id % 32) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
        }

        // Store
        __device__ __forceinline__ void Store(){
            int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
	    if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
                *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
	    }
        }
    };

    // 4 warps Tile_N = 128 16-bit 8-bit v=2 4
    template<typename OutType>
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
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b8b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
	    valid_tsize_ = 8 * vec_length; // =32/(8/vec_length)*2;
	    half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset; //32/(8/vec_length)
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
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

            int output_off = (intra_warp_tid_ % 4) * 2 + (intra_warp_tid_ / half_valid_tsize_) + (lane_id_ / 32) * 8;
	    if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off) = *(reinterpret_cast<int4 *>(output_fragment_));
	    }
        }
    };

    // 4 warps Tile_N = 64 16-bit 16-bit v=2 4
    template<typename OutType>
    struct wmmaOutputTile_16b{
        //
        // Member variables
        //

        int lane_id_;
        int valid_tsize_;
        int half_valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        unsigned long long* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
	    valid_tsize_ = 8 * vec_length; // =32/(8/vec_length)*2;
	    half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset; //32/(8/vec_length)
            output_matrix_ = reinterpret_cast<unsigned long long *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){
            int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);
            if(lane_id_ % 32 < half_valid_tsize_){
                for(int i = 0; i < 4; i++)
                    final_output_fragment_[i] += (final_output_fragment_[i+4] * 256);
	    }else if(lane_id_ % 32 < valid_tsize_){
                for(int i = 0; i < 4; i++)
                    final_output_fragment_[i] = final_output_fragment_[i] * 256 + final_output_fragment_[i+4] * 65536;
            }

            output_fragment_[(intra_warp_tid_/half_valid_tsize_+1)%2] = __shfl_xor_sync(0xffffffff, output_fragment_[(intra_warp_tid_/half_valid_tsize_+1)%2], half_valid_tsize_, 32);

	    if(lane_id_ % 32 < valid_tsize_){
	        for(int i = 0; i < 2; i++)
                    final_output_fragment_[i] += final_output_fragment_[i+2];
	    }

            int output_off = (intra_warp_tid_ % 4) * 2 + (intra_warp_tid_ / half_valid_tsize_) + (lane_id_ / 32) * 8;
	    if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off) = *(reinterpret_cast<unsigned long long *>(output_fragment_));
	    }
        }
    };

    // 4 warps Tile_N = 64 16-bit 16-bit v=8
    template<typename OutType>
    struct wmmaOutputTile_16b8v{
        //
        // Member variables
        //

        int lane_id_;
        int valid_tsize_;
        int half_valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
	    valid_tsize_ = 8 * vec_length; // =32/(8/vec_length)*2;
	    half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset; //32/(8/vec_length)
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){
	    //if(lane_id_ % 32 < valid_tsize_){
	    for(int i = 0; i < 8; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 256);

            for(int i = 0; i < 4; i++)
                output_fragment_0_[i] += (output_fragment_0_[i+4] * 256);
	    //}

            int output_off = intra_warp_tid_ % 4 + (lane_id_ / 32) * 4;
	    if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off) = *(reinterpret_cast<int4 *>(output_fragment_0_));
	    }
        }
    };

    // 4 warps Tile_N = 128 16-bit 8-bit v=8
    template<typename OutType>
    struct wmmaOutputTile_16b8b8v{
        //
        // Member variables
        //

        int lane_id_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b8b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
            const int output_offset = (m_index_vec * vec_length + (lane_id % 32) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
        }

        // Store
        __device__ __forceinline__ void Store(){
	    for(int i = 0; i < 8; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 256);

            int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 1);
	    
        }
    };

    //template<typename OutType>
    //struct wmmaOutputTile_8b8v{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile_8b8v(
    //        int lane_id,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        //vec_length = 4 ???
    //        const int output_offset = (m_index_vec * 4 + (lane_id % 32) / 4) * cols + column_offset;
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        lane_id_ = lane_id;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
    //        if(lane_id_ % 32 < 16){
    //            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
    //            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
    //        }
    //    }
    //};

    ////larger Tile_N = 64 8b4v
    //template<typename OutType>
    //struct wmmaOutputTile_8b4v{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile_8b4v(
    //        int lane_id,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        //vec_length = 4 ???
    //        const int output_offset = (m_index_vec * 4 + (lane_id % 32) / 4) * cols + column_offset;
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        lane_id_ = lane_id;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        int output_off = (lane_id_ % 4) * 2 + (lane_id_ / 32) * 8;
    //        if(lane_id_ % 32 < 16){
    //            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
    //            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
    //        }
    //    }
    //};

    // Tile_N = 128 4-bit 2 warps
    template<typename OutType>
    struct wmmaOutputTile_4b{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        // The register file fragment with the results to store
        int4* output_fragment_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_4b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<int4 *>(output_fragment);
	    valid_tsize_ = 4 * vec_length; // =32/(8/vec_length);
            const int output_offset = (m_index_vec * vec_length + (lane_id % 32) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
        }

        // Store
        __device__ __forceinline__ void Store(){
            int output_off = (lane_id_ % 4) * 4 + (lane_id_ / 32) * 16;
	    if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(output_fragment_ + 0);
                *(output_matrix_ + output_off + 1) = *(output_fragment_ + 1);
                *(output_matrix_ + output_off + 2) = *(output_fragment_ + 2);
                *(output_matrix_ + output_off + 3) = *(output_fragment_ + 3);
	    }
        }
    };

    template<typename OutType>
    struct wmmaOutputTile_8b4b8v{
        //
        // Member variables
        //
        int lane_id_;
        int half_valid_tsize_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b4b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
	    half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset; //32/(8/vec_length)
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
        }

        // Store
        __device__ __forceinline__ void Store(){

	    for(int i = 0; i < 16; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 16);

            int output_off = (lane_id_ % 4) * 4 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 1);
            *(output_matrix_ + output_off + 2) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 2);
            *(output_matrix_ + output_off + 3) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 3);
        }
    };

    template<typename OutType>
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
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b4b(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
	    valid_tsize_ = 8 * vec_length; // =32/(8/vec_length)*2;
	    half_valid_tsize_ = 4 * vec_length;
            const int output_offset = (m_index_vec * vec_length + (lane_id % half_valid_tsize_) / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 0], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 1], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 2], half_valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/half_valid_tsize_+1)%2)*4 + 3], half_valid_tsize_, 32);

            int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);
	    if(lane_id_ % 32 < valid_tsize_){
	        for(int i = 0; i < 8; i++)
                    final_output_fragment_[i] += (final_output_fragment_[i+8] * 16);
	    }

            int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / half_valid_tsize_) * 2 + (lane_id_ / 32) * 16;
	    if(lane_id_ % 32 < valid_tsize_){
                *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
                *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
	    }
        }
    };

    //template<typename OutType>
    //struct wmmaOutputTile_12b4b2v{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    int valid_tsize_;
    //    int intra_warp_tid_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile_12b4b2v(
    //        int lane_id, int vec_length,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        valid_tsize_ = 32 * vec_length / 8;
    //        const int output_offset = (m_index_vec * vec_length + (lane_id % valid_tsize_) / 4) * cols + column_offset; 
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        lane_id_ = lane_id;
    //        intra_warp_tid_ = lane_id % 32;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0], valid_tsize_, 32);
    //        output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1], valid_tsize_, 32);
    //        output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2], valid_tsize_, 32);
    //        output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3], valid_tsize_, 32);

    //        int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);
    //        if(intra_warp_tid_ < valid_tsize_*2){
    //            for(int i = 0; i < 8; i++)
    //                final_output_fragment_[i] += (final_output_fragment_[i+8] * 16);
    //        }else{
    //            for(int i = 0; i < 8; i++)
    //                final_output_fragment_[i] = final_output_fragment_[i] * 256;
    //        }
    //        output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 0], valid_tsize_*2, 32);
    //        output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 1], valid_tsize_*2, 32);

    //        for(int i = 0; i < 4; i++)
    //            final_output_fragment_[i] += final_output_fragment_[i+4];

    //        int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / valid_tsize_) % 2 * 2 + intra_warp_tid_ / valid_tsize_ / 2 + (lane_id_ / 32) * 16;
    //        *(output_matrix_ + output_off) = *(reinterpret_cast<int4 *>(output_fragment_));
    //    }
    //};

    template<typename OutType>
    struct wmmaOutputTile_12b4b2v{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_12b4b2v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
	    valid_tsize_ = 32 * vec_length / 8;
            const int output_offset = (m_index_vec * vec_length + (lane_id % valid_tsize_) / 4) * cols + column_offset; 
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3], valid_tsize_, 32);

            int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);
	    if(intra_warp_tid_ < valid_tsize_*2){
	        for(int i = 0; i < 8; i++)
                    final_output_fragment_[i] += (final_output_fragment_[i+8] * 16);
	    }else{
	        for(int i = 0; i < 8; i++)
                    final_output_fragment_[i] = final_output_fragment_[i] * 256 + final_output_fragment_[i+8] * 4096;
	    }
            output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 0], valid_tsize_*2, 32);
            output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/(valid_tsize_*2)+1)%2)*2 + 1], valid_tsize_*2, 32);

	    for(int i = 0; i < 4; i++)
                final_output_fragment_[i] += final_output_fragment_[i+4];

            int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / valid_tsize_) % 2 * 2 + intra_warp_tid_ / valid_tsize_ / 2 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off) = *(reinterpret_cast<int4 *>(output_fragment_));
        }
    };

    template<typename OutType>
    struct wmmaOutputTile_12b4b4v{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_12b4b4v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;

	    valid_tsize_ = 32 * vec_length / 8;
            const int output_offset = (m_index_vec * vec_length + (lane_id % valid_tsize_) / 4) * cols + column_offset; 
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){

	    if(intra_warp_tid_ < valid_tsize_){
	        for(int i = 0; i < 16; i++)
                    output_fragment_0_[i] += (output_fragment_1_[i] * 256);
	    }

            unsigned long long* output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment_0_);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3], valid_tsize_, 32);

	    for(int i = 0; i < 8; i++)
                output_fragment_0_[i] += (output_fragment_0_[i+8] * 16);

            int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / valid_tsize_) * 2 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 1);
        }
    };

    template<typename OutType>
    struct wmmaOutputTile_12b4b8v{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int* output_fragment_2_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_12b4b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            int* output_fragment_2,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
            output_fragment_2_ = output_fragment_2;

	    valid_tsize_ = 32 * vec_length / 8;
            const int output_offset = (m_index_vec * vec_length + (lane_id % valid_tsize_) / 4) * cols + column_offset; 
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){

	    for(int i = 0; i < 16; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 16 + output_fragment_2_[i] * 256);

            int output_off = (intra_warp_tid_ % 4) * 4 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 1);
            *(output_matrix_ + output_off + 2) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 2);
            *(output_matrix_ + output_off + 3) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 3);
        }
    };

    template<typename OutType>
    struct wmmaOutputTile_16b4b4v{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b4b4v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;

	    valid_tsize_ = 32 * vec_length / 8;
            const int output_offset = (m_index_vec * vec_length + (lane_id % valid_tsize_) / 4) * cols + column_offset; 
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){

	    if(intra_warp_tid_ < valid_tsize_){
	        for(int i = 0; i < 16; i++)
                    output_fragment_0_[i] += (output_fragment_1_[i] * 256);
	    }else{
	        for(int i = 0; i < 16; i++)
                    output_fragment_0_[i] = output_fragment_0_[i] * 16 + output_fragment_1_[i] * 4096;
	    }


            unsigned long long* output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment_0_);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 0], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 1], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 2], valid_tsize_, 32);
            output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/valid_tsize_+1)%2)*4 + 3], valid_tsize_, 32);

	    for(int i = 0; i < 8; i++)
                output_fragment_0_[i] += output_fragment_0_[i+8];

            int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / valid_tsize_) * 2 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 1);
        }
    };

    template<typename OutType>
    struct wmmaOutputTile_16b4b8v{
        //
        // Member variables
        //
        int lane_id_;
        int valid_tsize_;
        int intra_warp_tid_;
        // The register file fragment with the results to store
        int* output_fragment_0_;
        int* output_fragment_1_;
        int* output_fragment_2_;
        int* output_fragment_3_;
        int4* output_matrix_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b4b8v(
            int lane_id, int vec_length,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment_0,
            int* output_fragment_1,
            int* output_fragment_2,
            int* output_fragment_3,
            OutType* output_matrix)
        {
            output_fragment_0_ = output_fragment_0;
            output_fragment_1_ = output_fragment_1;
            output_fragment_2_ = output_fragment_2;
            output_fragment_3_ = output_fragment_3;

	    valid_tsize_ = 32 * vec_length / 8;
            const int output_offset = (m_index_vec * vec_length + (lane_id % valid_tsize_) / 4) * cols + column_offset; 
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
	    lane_id_ = lane_id;
            intra_warp_tid_ = lane_id % 32;
        }

        // Store
        __device__ __forceinline__ void Store(){

	    for(int i = 0; i < 16; i++)
                output_fragment_0_[i] += (output_fragment_1_[i] * 16 + output_fragment_2_[i] * 256 + output_fragment_3_[i] * 4096);

            int output_off = (intra_warp_tid_ % 4) * 4 + (lane_id_ / 32) * 16;
            *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 0);
            *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 1);
            *(output_matrix_ + output_off + 2) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 2);
            *(output_matrix_ + output_off + 3) = *(reinterpret_cast<int4 *>(output_fragment_0_) + 3);
        }
    };

    //larger Tile_N = 128
    template<typename OutType>
    struct wmmaOutputTile4_4bit{
        //
        // Member variables
        //
        int lane_id_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int4* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_int4;
        int thread_offset;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile4_4bit(
            int lane_id,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
            //const int output_offset = (m_index_vec * 4 + lane_id / 8 * 2) * cols + column_offset;
            const int output_offset = (m_index_vec * 4 + lane_id / 4) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
            rhs_columns_int4 = cols / 4;
            lane_id_ = lane_id;
	    //thread_offset = ((lane_id%8)%4/2 + (lane_id%8)%4%2*2)*2 + (lane_id%8)/4;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
            //output_fragment_[((lane_id_/2+1)%2)*4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4], 2, 32);
            //output_fragment_[((lane_id_/2+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 1], 2, 32);
            //output_fragment_[((lane_id_/2+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 2], 2, 32);
            //output_fragment_[((lane_id_/2+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 3], 2, 32);
            
            //output_fragment_[((lane_id_/4+1)%2)*2]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2], 4, 32);
            //output_fragment_[((lane_id_/4+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1], 4, 32);
            //output_fragment_[((lane_id_/4+1)%2)*2 + 4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 4], 4, 32);
            //output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4], 4, 32);

	    //if(lane_id_ < 16){
            //    #pragma unroll
            //    for (int i = 0; i < 4; i++){
            //        *(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
            //    }
	    //}
	    if(lane_id_ < 16){
                //#pragma unroll
                //for (int i = 0; i < 4; i++){
                //    *(output_matrix_ + (lane_id_ % 4) * 4 + i) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
                //    //*(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
                //    //*(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + (lane_id_%8)) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
                //}
                *(output_matrix_ + (lane_id_ % 4) * 4 + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
                *(output_matrix_ + (lane_id_ % 4) * 4 + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
                *(output_matrix_ + (lane_id_ % 4) * 4 + 2) = *(reinterpret_cast<int4 *>(output_fragment_) + 2);
                *(output_matrix_ + (lane_id_ % 4) * 4 + 3) = *(reinterpret_cast<int4 *>(output_fragment_) + 3);

                *(output_matrix_ + (lane_id_ % 4) * 4 + 16) = *(reinterpret_cast<int4 *>(output_fragment_) + 4);
                *(output_matrix_ + (lane_id_ % 4) * 4 + 17) = *(reinterpret_cast<int4 *>(output_fragment_) + 5);
                *(output_matrix_ + (lane_id_ % 4) * 4 + 18) = *(reinterpret_cast<int4 *>(output_fragment_) + 6);
                *(output_matrix_ + (lane_id_ % 4) * 4 + 19) = *(reinterpret_cast<int4 *>(output_fragment_) + 7);
	    }
        }
    };

    ////larger Tile_N = 128 4b8v
    //template<typename OutType>
    //struct wmmaOutputTile_4b8v{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;
    //    // The number of columns in the rhs matrix
    //    //int rhs_columns_int4;
    //    //int thread_offset;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile_4b8v(
    //        int lane_id,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        //vec_length = 8 ???
    //        const int output_offset = (m_index_vec * 8 + (lane_id % 32) / 4) * cols + column_offset;
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        lane_id_ = lane_id;
    //        //rhs_columns_int4 = cols / 4;
    //        //thread_offset = ((lane_id%8)%4/2 + (lane_id%8)%4%2*2)*2 + (lane_id%8)/4;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        // Step 1: warp shuffle to align the memory access
    //        //output_fragment_[((lane_id_/2+1)%2)*4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 1], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 2], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 3], 2, 32);
    //        
    //        //output_fragment_[((lane_id_/4+1)%2)*2]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 4], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4], 4, 32);

    //        //if(lane_id_ < 16){
    //        //    #pragma unroll
    //        //    for (int i = 0; i < 4; i++){
    //        //        *(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //        //    }
    //        //}
    //        int output_off = ((lane_id_ % 32) % 4) * 4 + (lane_id_ / 32) * 16;
    //        *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
    //        *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
    //        *(output_matrix_ + output_off + 2) = *(reinterpret_cast<int4 *>(output_fragment_) + 2);
    //        *(output_matrix_ + output_off + 3) = *(reinterpret_cast<int4 *>(output_fragment_) + 3);
    //    }
    //};

    //template<typename OutType>
    //struct wmmaOutputTile_8b4b4v{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    int intra_warp_tid_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile_8b4b4v(
    //        int lane_id,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        const int output_offset = (m_index_vec * 4 + (lane_id % 16) / 4) * cols + column_offset;
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        lane_id_ = lane_id;
    //        intra_warp_tid_ = lane_id % 32;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 0] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 0], 16, 32);
    //        output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 1], 16, 32);
    //        output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 2], 16, 32);
    //        output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((intra_warp_tid_/16+1)%2)*4 + 3], 16, 32);

    //        int *final_output_fragment_ = reinterpret_cast<int *>(output_fragment_);
    //        for(int i = 0; i < 8; i++)
    //            final_output_fragment_[i] += (final_output_fragment_[i+8] * 16);

    //        int output_off = (intra_warp_tid_ % 4) * 4 + (intra_warp_tid_ / 16) * 2 + (lane_id_ / 32) * 16;
    //        *(output_matrix_ + output_off + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
    //        *(output_matrix_ + output_off + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
    //    }
    //};

    ////larger Tile_N = 128
    //template<typename OutType>
    //struct wmmaOutputTile4_4bit{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;
    //    // The number of columns in the rhs matrix
    //    int rhs_columns_int4;
    //    int thread_offset;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile4_4bit(
    //        int lane_id,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        //const int output_offset = (m_index_vec * 4 + lane_id / 8 * 2) * cols + column_offset;
    //        const int output_offset = (m_index_vec * 4 + lane_id / 4) * cols + column_offset;
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        rhs_columns_int4 = cols / 4;
    //        lane_id_ = lane_id;
    //        //thread_offset = ((lane_id%8)%4/2 + (lane_id%8)%4%2*2)*2 + (lane_id%8)/4;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        // Step 1: warp shuffle to align the memory access
    //        //output_fragment_[((lane_id_/2+1)%2)*4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 1], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 2], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 3], 2, 32);
    //        
    //        //output_fragment_[((lane_id_/4+1)%2)*2]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 4], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4], 4, 32);

    //        //if(lane_id_ < 16){
    //        //    #pragma unroll
    //        //    for (int i = 0; i < 4; i++){
    //        //        *(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //        //    }
    //        //}
    //        if(lane_id_ < 16){
    //            //#pragma unroll
    //            //for (int i = 0; i < 4; i++){
    //            //    *(output_matrix_ + (lane_id_ % 4) * 4 + i) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //            //    //*(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //            //    //*(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + (lane_id_%8)) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //            //}
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 2) = *(reinterpret_cast<int4 *>(output_fragment_) + 2);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 3) = *(reinterpret_cast<int4 *>(output_fragment_) + 3);

    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 16) = *(reinterpret_cast<int4 *>(output_fragment_) + 4);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 17) = *(reinterpret_cast<int4 *>(output_fragment_) + 5);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 18) = *(reinterpret_cast<int4 *>(output_fragment_) + 6);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 19) = *(reinterpret_cast<int4 *>(output_fragment_) + 7);
    //        }
    //    }
    //};

    //template<typename OutType>
    //struct wmmaOutputTile4_4bit{
    //    //
    //    // Member variables
    //    //
    //    int lane_id_;
    //    // The register file fragment with the results to store
    //    unsigned long long* output_fragment_;
    //    int4* output_matrix_;
    //    // The number of columns in the rhs matrix
    //    int rhs_columns_int4;
    //    int thread_offset;

    //    // Constructor
    //    __device__ __forceinline__ wmmaOutputTile4_4bit(
    //        int lane_id,
    //        int m_index_vec, int column_offset,
    //        int cols,
    //        int* output_fragment,
    //        OutType* output_matrix)
    //    {
    //        output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
    //        //const int output_offset = (m_index_vec * 4 + lane_id / 8 * 2) * cols + column_offset;
    //        const int output_offset = (m_index_vec * 4 + lane_id / 4) * cols + column_offset;
    //        output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
    //        rhs_columns_int4 = cols / 4;
    //        lane_id_ = lane_id;
    //        //thread_offset = ((lane_id%8)%4/2 + (lane_id%8)%4%2*2)*2 + (lane_id%8)/4;
    //    }

    //    // Store
    //    __device__ __forceinline__ void Store(){
    //        // Step 1: warp shuffle to align the memory access
    //        //output_fragment_[((lane_id_/2+1)%2)*4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 1], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 2] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 2], 2, 32);
    //        //output_fragment_[((lane_id_/2+1)%2)*4 + 3] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/2+1)%2)*4 + 3], 2, 32);
    //        
    //        //output_fragment_[((lane_id_/4+1)%2)*2]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 4]     = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 4], 4, 32);
    //        //output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4], 4, 32);

    //        //if(lane_id_ < 16){
    //        //    #pragma unroll
    //        //    for (int i = 0; i < 4; i++){
    //        //        *(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //        //    }
    //        //}
    //        if(lane_id_ < 16){
    //            //#pragma unroll
    //            //for (int i = 0; i < 4; i++){
    //            //    *(output_matrix_ + (lane_id_ % 4) * 4 + i) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //            //    //*(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + thread_offset) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //            //    //*(output_matrix_ + i % 2 * rhs_columns_int4 + (i/2) * 8 + (lane_id_%8)) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
    //            //}
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 0) = *(reinterpret_cast<int4 *>(output_fragment_) + 0);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 1) = *(reinterpret_cast<int4 *>(output_fragment_) + 1);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 2) = *(reinterpret_cast<int4 *>(output_fragment_) + 2);
    //            *(output_matrix_ + (lane_id_ % 4) * 4 + 3) = *(reinterpret_cast<int4 *>(output_fragment_) + 3);
    //        }
    //    }
    //};

    template<typename OutType>
    struct wmmaOutputTile4_8bit{
        //
        // Member variables
        //
        int lane_id_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int4* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_int4;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile4_8bit(
            int lane_id,
            int m_index_vec, int column_offset,
            int cols,
            int* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
            const int output_offset = (m_index_vec * 4 + lane_id / 8 * 2) * cols + column_offset;
            output_matrix_ = reinterpret_cast<int4 *>(output_matrix + output_offset);
            rhs_columns_int4 = cols / 4;
            lane_id_ = lane_id;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
             
            //#pragma unroll
            //for (int i = 1; i < 5; i*=2){
            //    output_fragment_[((lane_id_/i+1)%2)*2] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/i+1)%2)*2], i, 32);
            //    output_fragment_[((lane_id_/i+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/i+1)%2)*2 + 1], i, 32);

            //    output_fragment_[((lane_id_/i+1)%2)*2 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/i+1)%2)*2 + 4], i, 32);
            //    output_fragment_[((lane_id_/i+1)%2)*2 + 1 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/i+1)%2)*2 + 1 + 4], i, 32);
            //}

	    //if(lane_id_ < 16){
            //    #pragma unroll
            //    for (int i = 0; i < 4; i++){
            //        *(output_matrix_ + i % 2 * rhs_columns_int4 + 8 * (i/2) + lane_id_%8) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
            //    }
	    //}
	    
	    
            output_fragment_[((lane_id_/4+1)%2)*2] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2], 4, 32);
            output_fragment_[((lane_id_/4+1)%2)*2 + 1] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1], 4, 32);

            output_fragment_[((lane_id_/4+1)%2)*2 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 4], 4, 32);
            output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4] = __shfl_xor_sync(0xffffffff, output_fragment_[((lane_id_/4+1)%2)*2 + 1 + 4], 4, 32);
            

	    if(lane_id_ < 16){
                #pragma unroll
                for (int i = 0; i < 4; i++){
                    *(output_matrix_ + i % 2 * rhs_columns_int4 + 8 * (i/2) + (lane_id_%8)%4*2+(lane_id_%8)/4) = *(reinterpret_cast<int4 *>(output_fragment_) + i);
                }
	    }
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
            int m_index_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float *>(output_fragment);
            const int output_offset = (m_index_vec * 4) * cols + column_offset + thread_group * 8 + lane_id;
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
                *(output_matrix_ + i * rhs_columns_) = OutType(*(output_fragment_ + i));
                *(output_matrix_ + 4 + i * rhs_columns_) = OutType(*(output_fragment_ + 4 + i));
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
            int m_index_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float *>(output_fragment);
            const int output_offset = (m_index_vec * 2) * cols + column_offset + thread_group * 8 + lane_id;
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
                *(output_matrix_ + i * rhs_columns_) = OutType(*(output_fragment_ + i));
                *(output_matrix_ + 4 + i * rhs_columns_) = OutType(*(output_fragment_ + 2 + i));
            }
        }
    };
}
#endif
