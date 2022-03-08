namespace sddmm{
    template <int ValuesBlockWidth, int Blocks>
    struct wmmaComputeUtils_4b{

        // Shared memory buffers
        const int* lhs_tile_;
        int* rhs_prefetch_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;
        int warp_id_;
        int active_warp_num_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_4b(
	    int lane_id,
	    int workset,
            const int* lhs_tile,
            int* rhs_prefetch,
            int* output_fragment):
            lane_id_(lane_id),
            warp_id_(lane_id / 32),
            active_warp_num_((workset + 8 - 1) / 8),
            lhs_tile_(lhs_tile),
            rhs_prefetch_(rhs_prefetch),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[Blocks] = {};

            int *rhs_fragment = rhs_prefetch_;
            if(warp_id_ < active_warp_num_){
	        if(lane_id_ % 32 < ValuesBlockWidth){
                    #pragma unroll
	            for(int i=0; i<Blocks; i++){
	                lhs_fragment[i] = lhs_tile_[i*ValuesBlockWidth + lane_id_ % 32 + (step % 2) * ValuesBlockWidth * Blocks];
	            }
	        }

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                        "r"(lhs_fragment[i]),
                        "r"(rhs_fragment[i])
                    );
                }
	    }

        }

        //__device__ __forceinline__ void TileMACResidue(){
	//    printf("Residue not supported\n");
        //}
    };
}
