#ifndef SPMM_COMPUTE_UTILS_H
#define SPMM_COMPUTE_UTILS_H

namespace spmm{
    template <typename VecType, int Tile_N, int Tile_K, int BlockWidth, int VecLength>
    struct ComputeUtils {

        //
        // Static membrs
        //

        static constexpr int kThreadItemsK = Tile_K / BlockWidth;

        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values.
        const VecType* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        __device__ __forceinline__ ComputeUtils(
            const half* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment):
            lhs_tile_(reinterpret_cast<const VecType*>(lhs_tile)),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}
        

        __device__ __forceinline__ void TileMAC(){
            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < Tile_N; n_item_idx ++){
                half lhs_value[VecLength];
                VecType * lhs_value_v = reinterpret_cast<VecType *>(lhs_value);
                *(lhs_value_v) = *(lhs_tile_ + n_item_idx);
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; k_item_idx ++){
                    half rhs_value = *(rhs_fragment_ + kThreadItemsK * n_item_idx + k_item_idx);
                    #pragma unroll
                    for (int v = 0; v < VecLength; v++){
                        *(output_fragment_ + k_item_idx + v * kThreadItemsK) += __half2float(lhs_value[v] * rhs_value);
                    }
                }
            }
        }

    };

    template <typename VecType, int Tile_N, int Tile_K, int BlockWidth, int VecLength>
    struct ComputeUtils1D {

        //
        // Static membrs
        //

        static constexpr int kThreadItemsK = Tile_K / BlockWidth;

        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values.
        const VecType* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        __device__ __forceinline__ ComputeUtils1D(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment):
            lhs_tile_(lhs_tile),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}
        

        __device__ __forceinline__ void TileMAC(){
            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < Tile_N; n_item_idx ++){
                half lhs_value[VecLength];
                VecType * lhs_value_v = reinterpret_cast<VecType *>(lhs_value);
                *(lhs_value_v) = *(lhs_tile_ + n_item_idx);
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; k_item_idx ++){
                    half rhs_value = *(rhs_fragment_ + kThreadItemsK * n_item_idx + k_item_idx);
                    #pragma unroll
                    for (int v = 0; v < VecLength; v++){
                        *(output_fragment_ + k_item_idx + v * kThreadItemsK) += __half2float(lhs_value[v] * rhs_value);
                    }
                }
            }
        }

    };

    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils8 {

        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 4 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const float2* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils8(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}
        

        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
            #pragma unroll
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];
            
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);
            
            #pragma unroll
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                );
            }
        }
    };

    // Compute Tile for k=4, 8-bit integer
    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils4_8bit {
        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 16 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils4_8bit(
            const VecType* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 72 + lane_id_/4 + n_group_idx * 72 * 4;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[4+i] = *(dense_tile_ + base_offset + 8 + i*16); 
	    }

            char *rhs_fragment_char = reinterpret_cast<char *>(rhs_fragment); 
            char *rhs_fragment_transpose_char = reinterpret_cast<char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	            *(rhs_fragment_transpose_char + 16 + j*4 + i) = *(rhs_fragment_char + 16 + j + i*4);
		}
	    }
            
	    if(lane_id_ < 16){
	        lhs_fragment[0] = lhs_tile_[lane_id_+n_group_idx*16];
	    }
	    else{
	        lhs_fragment[0] = 0;
	    } 

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[8 + i]), "+r"(output_fragment_[12 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i+4])
                );
            }
        }

        //// Compute Residue
        //__device__ __forceinline__ void TileMACResidue(int n_group_idx){
        //    int lhs_fragment[2];
        //    int waste[4];
        //    float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
        //    if (thread_group_ < 4) *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 4);
        //    int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
        //    const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

        //    #pragma unroll
        //    for (int i = 0; i < 8; i ++){
        //        asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
        //            "{%0, %1}, \t"
        //            "{%2}, \t"
        //            "{%3}, \t"
        //            "{%0, %1}; ":
        //            "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[1 + 2 * i]),
        //            "r"(rhs_fragment_int[i]),
        //            "r"(lhs_fragment_int[0])
        //        );
        //    }
        //}
    };

    // Compute Tile for k=4
    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils4 {
        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 4 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const float2* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into
        float* output_fragment_;
        int thread_group_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils4(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment),
            thread_group_(thread_group){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            float waste[4];
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            if (thread_group_ < 4) *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 4);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];
            float waste[4];
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            if (thread_group_ < 4) *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 4);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };

    // Compute Tile for k=2
    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils2 {
        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_N / 4 - 1;

        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile value
        const float* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        int thread_group_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils2(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float *>(lhs_tile) + lane_id),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment),
            thread_group_(thread_group){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            float waste[6];
            if (thread_group_ < 4) *(lhs_fragment) = *(lhs_tile_ + n_group_idx * 4);
            half* lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
            *(lhs_fragment_half + 2) = *(lhs_fragment_half + 1);

            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[1 + 2 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]), "+f"(waste[4]), "+f"(waste[5]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];
            float waste[6];
            if (thread_group_ < 4) *(lhs_fragment) = *(lhs_tile_ + n_group_idx * 4);
            half* lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
            *(lhs_fragment_half + 2) = *(lhs_fragment_half + 1);

            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[1 + 2 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]), "+f"(waste[4]), "+f"(waste[5]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };
}

#endif
