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

    template <int ValuesBlockWidth, int Blocks>
    struct wmmaComputeUtils_8b{

        // Shared memory buffers
        const int* lhs_tile_;
        int* rhs_prefetch_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;
        int warp_id_;
        int active_warp_num_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_8b(
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
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
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

    template <int ValuesBlockWidth, int Blocks>
    struct wmmaComputeUtils_16b{

        // Shared memory buffers
        const int* lhs_tile_;
        unsigned char* rhs_prefetch_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;
        int warp_id_;
        int active_warp_num_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b(
	    int lane_id,
	    int workset,
            const int* lhs_tile,
            int* rhs_prefetch,
            int* output_fragment):
            lane_id_(lane_id),
            warp_id_(lane_id / 32),
            active_warp_num_((workset + 8 - 1) / 8),
            lhs_tile_(lhs_tile),
            rhs_prefetch_(reinterpret_cast<unsigned char *>(rhs_prefetch)),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[Blocks*2] = {};
            int lhs_fragment_decompose[Blocks] = {};
            int rhs_fragment_decompose[Blocks*2] = {};

            if(warp_id_ < active_warp_num_){
	        if(lane_id_ % 32 < ValuesBlockWidth){
                    #pragma unroll
	            for(int i=0; i<Blocks; i++){
	                lhs_fragment[i*2 + 0] = lhs_tile_[i*ValuesBlockWidth + lane_id_ % (ValuesBlockWidth/2)*2 + (step % 2) * ValuesBlockWidth * Blocks];
	                lhs_fragment[i*2 + 1] = lhs_tile_[i*ValuesBlockWidth + lane_id_ % (ValuesBlockWidth/2)*2 + 1 + (step % 2) * ValuesBlockWidth * Blocks];
	            }
	        }
		
		unsigned char *lhs_fragment_char = reinterpret_cast<unsigned char *>(lhs_fragment);
		unsigned char *lhs_fragment_decompose_char = reinterpret_cast<unsigned char *>(lhs_fragment_decompose);
		unsigned char *rhs_fragment_decompose_char = reinterpret_cast<unsigned char *>(rhs_fragment_decompose);

                if(lane_id_ % 32 < ValuesBlockWidth){
		    if(lane_id_ % 32 < ValuesBlockWidth/2){
                        #pragma unroll
			for(int i=0; i<Blocks*4; i++)
			    lhs_fragment_decompose_char[i] = lhs_fragment_char[i*2];
		    }
		    else{
                        #pragma unroll
			for(int i=0; i<Blocks*4; i++)
			    lhs_fragment_decompose_char[i] = lhs_fragment_char[i*2 + 1];
		    }
		}

                #pragma unroll
		for(int i=0; i<Blocks*8; i++)
	            rhs_fragment_decompose_char[i/2+i%2*Blocks*4] = rhs_prefetch_[i];

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                        "r"(lhs_fragment_decompose[i]),
                        "r"(rhs_fragment_decompose[i])
                    );
                }

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[2]), "+r"(output_fragment_[3]):
                        "r"(lhs_fragment_decompose[i]),
                        "r"(rhs_fragment_decompose[i+Blocks])
                    );
                }
	    }

        }

        //__device__ __forceinline__ void TileMACResidue(){
	//    printf("Residue not supported\n");
        //}
    };

    template <int ValuesBlockWidth, int Blocks>
    struct wmmaComputeUtils_16b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        unsigned char* rhs_prefetch_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;
        int warp_id_;
        int active_warp_num_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8v(
	    int lane_id,
	    int workset,
            const int* lhs_tile,
            int* rhs_prefetch,
            int* output_fragment):
            lane_id_(lane_id),
            warp_id_(lane_id / 32),
            active_warp_num_((workset + 8 - 1) / 8),
            lhs_tile_(lhs_tile),
            rhs_prefetch_(reinterpret_cast<unsigned char *>(rhs_prefetch)),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[Blocks*2] = {};
            int lhs_fragment_decompose[Blocks*2] = {};
            int rhs_fragment_decompose[Blocks*2] = {};

            if(warp_id_ < active_warp_num_){
	        if(lane_id_ % 32 < ValuesBlockWidth){
                    #pragma unroll
	            for(int i=0; i<Blocks; i++){
	                //lhs_fragment[i*2 + 0] = lhs_tile_[i*(ValuesBlockWidth+2) + lane_id_ % (ValuesBlockWidth/2)*2 + lane_id_ % (ValuesBlockWidth/2)/16 + (step % 2) * (ValuesBlockWidth * Blocks + 4)];
	                //lhs_fragment[i*2 + 1] = lhs_tile_[i*(ValuesBlockWidth+2) + lane_id_ % (ValuesBlockWidth/2)*2 + lane_id_ % (ValuesBlockWidth/2)/16 + 1 + (step % 2) * (ValuesBlockWidth * Blocks + 4)];
	                lhs_fragment[i*2 + 0] = lhs_tile_[i*ValuesBlockWidth + lane_id_ % (ValuesBlockWidth/2)*2 + (step % 2) * ValuesBlockWidth * Blocks];
	                lhs_fragment[i*2 + 1] = lhs_tile_[i*ValuesBlockWidth + lane_id_ % (ValuesBlockWidth/2)*2 + 1 + (step % 2) * ValuesBlockWidth * Blocks];
	            }
	        }
		
		unsigned char *lhs_fragment_char = reinterpret_cast<unsigned char *>(lhs_fragment);
		unsigned char *lhs_fragment_decompose_char = reinterpret_cast<unsigned char *>(lhs_fragment_decompose);
		unsigned char *rhs_fragment_decompose_char = reinterpret_cast<unsigned char *>(rhs_fragment_decompose);

                #pragma unroll
		for(int i=0; i<Blocks*8; i++){
	            lhs_fragment_decompose_char[i/2+i%2*Blocks*4] = lhs_fragment_char[i];
	            rhs_fragment_decompose_char[i/2+i%2*Blocks*4] = rhs_prefetch_[i];
	        }

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
                        "r"(lhs_fragment_decompose[i]),
                        "r"(rhs_fragment_decompose[i])
                    );
                }

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[2]), "+r"(output_fragment_[3]):
                        "r"(lhs_fragment_decompose[i]),
                        "r"(rhs_fragment_decompose[i+Blocks])
                    );
                }

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[4]), "+r"(output_fragment_[5]):
                        "r"(lhs_fragment_decompose[i+Blocks]),
                        "r"(rhs_fragment_decompose[i])
                    );
                }

                #pragma unroll
                for (int i=0; i<Blocks; i++){
                    asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                        "{%0, %1}, \t"
                        "{%2}, \t"
                        "{%3}, \t"
                        "{%0, %1}; ":
                        "+r"(output_fragment_[6]), "+r"(output_fragment_[7]):
                        "r"(lhs_fragment_decompose[i+Blocks]),
                        "r"(rhs_fragment_decompose[i+Blocks])
                    );
                }
	    }

        }

        //__device__ __forceinline__ void TileMACResidue(){
	//    printf("Residue not supported\n");
        //}
    };
