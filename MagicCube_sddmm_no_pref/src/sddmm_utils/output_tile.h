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
}
