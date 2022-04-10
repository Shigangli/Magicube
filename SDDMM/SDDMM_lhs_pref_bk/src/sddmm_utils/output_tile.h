namespace sddmm{
    template<int VecLength>
    struct wmmaOutputTile_4b{
        //
        // Member variables
        //
        int lane_id_;
        int warp_id_;
        int active_warp_num_;
        int active_thread_num_;
        int output_warp_width_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int* output_values_base_;
        // The number of columns in the rhs matrix

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_4b(
            int lane_id,
            int offset, 
	    int workset,
            int* output_fragment,
            int* output_values)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
            output_values_base_ = output_values + offset*VecLength;
            active_warp_num_ = (workset + 8 - 1) / 8,
            active_thread_num_ = 32 * VecLength / 8;
	    warp_id_ = lane_id / 32;
	    lane_id_ = lane_id;
	    output_warp_width_ = VecLength * 8;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if(warp_id_ < active_warp_num_){
	        if(lane_id_ % 32 < active_thread_num_)
		    *(reinterpret_cast<unsigned long long *>(output_values_base_ + (lane_id_ % 32) * 2 + warp_id_ * output_warp_width_)) = output_fragment_[0];
	    }
        }
    };

    template<int VecLength>
    struct wmmaOutputTile_8b{
        //
        // Member variables
        //
        int lane_id_;
        int warp_id_;
        int active_warp_num_;
        int active_thread_num_;
        int output_warp_width_;
        // The register file fragment with the results to store
        unsigned long long* output_fragment_;
        int* output_values_base_;
        // The number of columns in the rhs matrix

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_8b(
            int lane_id,
            int offset, 
	    int workset,
            int* output_fragment,
            int* output_values)
        {
            output_fragment_ = reinterpret_cast<unsigned long long *>(output_fragment);
            output_values_base_ = output_values + offset*VecLength;
            active_warp_num_ = (workset + 8 - 1) / 8,
            active_thread_num_ = 32 * VecLength / 8;
	    warp_id_ = lane_id / 32;
	    lane_id_ = lane_id;
	    output_warp_width_ = VecLength * 8;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if(warp_id_ < active_warp_num_){
	        if(lane_id_ % 32 < active_thread_num_)
		    *(reinterpret_cast<unsigned long long *>(output_values_base_ + (lane_id_ % 32) * 2 + warp_id_ * output_warp_width_)) = output_fragment_[0];
	    }
        }
    };

    template<int VecLength>
    struct wmmaOutputTile_16b{
        //
        // Member variables
        //
        int lane_id_;
        int warp_id_;
        int active_warp_num_;
        int active_thread_num_;
        int output_warp_width_;
        int half_active_thread_num_;
	int intra_warp_tid_;

        // The register file fragment with the results to store
        int* output_fragment_;
        int* output_values_base_;
        // The number of columns in the rhs matrix

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b(
            int lane_id,
            int offset, 
	    int workset,
            int* output_fragment,
            int* output_values)
        {
	    output_fragment_ = output_fragment;
            output_values_base_ = output_values + offset*VecLength;
            active_warp_num_ = (workset + 8 - 1) / 8,
            active_thread_num_ = 32 * VecLength / 8 * 2; //shuffle x2
            half_active_thread_num_ = 32 * VecLength / 8; 
	    warp_id_ = lane_id / 32;
	    intra_warp_tid_ = lane_id % 32;
	    lane_id_ = lane_id;
	    output_warp_width_ = VecLength * 8;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if(warp_id_ < active_warp_num_){
		for(int i = 0; i < 2; i++){
		    output_fragment_[i] += 256 * output_fragment_[i+2];
		}
                
		output_fragment_[(intra_warp_tid_/half_active_thread_num_+1)%2] = __shfl_xor_sync(0xffffffff, output_fragment_[(intra_warp_tid_/half_active_thread_num_+1)%2], half_active_thread_num_, 32);

		output_fragment_[0] += 256 * output_fragment_[1];

	        if(intra_warp_tid_ < active_thread_num_)
		    *(output_values_base_ + (intra_warp_tid_ % half_active_thread_num_) * 2 + intra_warp_tid_ / half_active_thread_num_ + warp_id_ * output_warp_width_) = output_fragment_[0];
	    }
        }
    };

    template<int VecLength>
    struct wmmaOutputTile_16b8v{
        //
        // Member variables
        //
        int lane_id_;
        int warp_id_;
        int active_warp_num_;
        int output_warp_width_;
	int intra_warp_tid_;

        // The register file fragment with the results to store
        int* output_fragment_;
        int* output_values_base_;
        // The number of columns in the rhs matrix

        // Constructor
        __device__ __forceinline__ wmmaOutputTile_16b8v(
            int lane_id,
            int offset, 
	    int workset,
            int* output_fragment,
            int* output_values)
        {
	    output_fragment_ = output_fragment;
            output_values_base_ = output_values + offset*VecLength;
            active_warp_num_ = (workset + 8 - 1) / 8,
	    warp_id_ = lane_id / 32;
	    intra_warp_tid_ = lane_id % 32;
	    lane_id_ = lane_id;
	    output_warp_width_ = VecLength * 8;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if(warp_id_ < active_warp_num_){
		for(int i = 0; i < 2; i++){
		    output_fragment_[i] += (256 * output_fragment_[i+2] + 256 * output_fragment_[i+4] + 256 * 256 * output_fragment_[i+6]);
		}

		*(reinterpret_cast<unsigned long long *>(output_values_base_ + intra_warp_tid_ * 2 + warp_id_ * output_warp_width_)) = *(reinterpret_cast<unsigned long long *>(output_fragment_));
	    }
        }
    };
}
