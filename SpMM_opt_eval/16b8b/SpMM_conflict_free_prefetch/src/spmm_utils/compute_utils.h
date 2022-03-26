#ifndef SPMM_COMPUTE_UTILS_H
#define SPMM_COMPUTE_UTILS_H

namespace spmm{
    template <typename VecType, int Tile_K, int Tile_N, int BlockWidth, int VecLength>
    struct ComputeUtils {

        //
        // Static membrs
        //

        static constexpr int kThreadItemsK = Tile_N / BlockWidth;

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
            for (int n_item_idx = 0; n_item_idx < Tile_K; n_item_idx ++){
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

    template <typename VecType, int Tile_K, int Tile_N, int BlockWidth, int VecLength>
    struct ComputeUtils1D {

        //
        // Static membrs
        //

        static constexpr int kThreadItemsK = Tile_N / BlockWidth;

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
            for (int n_item_idx = 0; n_item_idx < Tile_K; n_item_idx ++){
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

    template <typename VecType, int Tile_K>
    struct wmmaComputeUtils8 {

        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_K / 4 - 1;
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

    //template <int Tile_K>
    //struct wmmaComputeUtils4_4bit {

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils4_4bit(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int n_group_idx){
    //        int lhs_fragment[1];
    //        int rhs_fragment[8];
    //        int rhs_fragment_transpose[8];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + lane_id_/4;
    //        //int base_offset = chunk_id * 72 + lane_id_/4 + n_group_idx * 72 * 4;
    //        //int base_offset = chunk_id * 64 + lane_id_/4 + n_group_idx * 64 * 4;

    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*8); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        //baseline with more bit-wise shift
    //        //unsigned char maskc = 0x0F;
    //        //#pragma unroll
    //        //for(int j = 0; j < 8; j++)
    //        //    for(int v = 0; v < 4; v++){
    //        //	    int intra_char_offset_0 = (j%2)*4;
    //        //	    int intra_char_offset_1 = ((j+1)%2)*4;
    //        //        rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
    //        //        rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
    //        //    }

    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }
    //        unsigned int mask0 = 0xF0F0F0F0;
    //        unsigned int mask1 = 0x0F0F0F0F;
    //        unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
    //        unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

    //        //#pragma unroll
    //        //for(int i=0; i<4; i++){
    //        //    rhs_fragment_uint[i*2] = (rhs_fragment_transpose_uint[i*2] & mask1) | ((rhs_fragment_transpose_uint[i*2+1] & mask1) << 4);
    //        //    rhs_fragment_uint[i*2+1] = ((rhs_fragment_transpose_uint[i*2] & mask0) >> 4) | (rhs_fragment_transpose_uint[i*2+1] & mask0);
    //        //}
    //        
    //        rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
    //        rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
    //        rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
    //        rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
    //        rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
    //        rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
    //        rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
    //        rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

    //        if(lane_id_ < 16){
    //            lhs_fragment[0] = lhs_tile_[lane_id_+n_group_idx*16];
    //        }
    //        else{
    //            lhs_fragment[0] = 0;
    //        } 

    //        #pragma unroll
    //        for (int i = 0; i < 8; i ++){
    //            asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment[i])
    //            );
    //        }
    //    }
    //};

    //template <int Tile_K>
    //struct wmmaComputeUtils_8b4b4v{

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils_8b4b4v(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int step){
    //        int lhs_fragment[1];
    //        int rhs_fragment[8];
    //        int rhs_fragment_transpose[8];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 136 + lane_id_ / 4;

    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        unsigned int mask0 = 0xF0F0F0F0;
    //        unsigned int mask1 = 0x0F0F0F0F;
    //        unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
    //        unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
    //        
    //        rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
    //        rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
    //        rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
    //        rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
    //        rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
    //        rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
    //        rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
    //        rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

    //        lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * 32];
    //        #pragma unroll
    //        for (int i = 0; i < 8; i ++){
    //            asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment[i])
    //            );
    //        }
    //    }

    //    __device__ __forceinline__ void TileMACResidue(){
    //        int lhs_fragment[1];
    //        int rhs_fragment[8];
    //        int rhs_fragment_transpose[8];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 136 + lane_id_ / 4;
    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }
    //        unsigned int mask0 = 0xF0F0F0F0;
    //        unsigned int mask1 = 0x0F0F0F0F;
    //        unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
    //        unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

    //        rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
    //        rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
    //        rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
    //        rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
    //        rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
    //        rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
    //        rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
    //        rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

    //        lhs_fragment[0] = lhs_tile_[lane_id_%32];
    //        #pragma unroll
    //        for (int i = 0; i < 8; i ++){
    //            asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment[i])
    //            );
    //        }
    //    }
    //};

    //// Tile_N=128 4bit v=8
    //template <int Tile_K>
    //struct wmmaComputeUtils_4b8v{

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils_4b8v(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int step){
    //        int lhs_fragment[1];
    //        int rhs_fragment[8];
    //        //int rhs_fragment[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    //        int rhs_fragment_transpose[8];
    //        int chunk_id = lane_id_ % 4;
    //        //int base_offset = chunk_id * 136 + (lane_id_ % 32 )/4 + (lane_id_ / 32) * 8;
    //        int base_offset = chunk_id * 136 + lane_id_ / 4;
    //        //int base_offset = chunk_id * 72 + lane_id_/4 + n_group_idx * 72 * 4;
    //        //int base_offset = chunk_id * 64 + lane_id_/4 + n_group_idx * 64 * 4;

    //        //char index_c[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //            //rhs_fragment[index_c[i]*2] = *(dense_tile_ + base_offset + index_c[i]*16); 
    //            //rhs_fragment[index_c[i]*2+1] = *(dense_tile_ + base_offset + 8 + index_c[i]*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        //baseline with more bit-wise shift
    //        //unsigned char maskc = 0x0F;
    //        //#pragma unroll
    //        //for(int j = 0; j < 8; j++)
    //        //    for(int v = 0; v < 4; v++){
    //        //	    int intra_char_offset_0 = (j%2)*4;
    //        //	    int intra_char_offset_1 = ((j+1)%2)*4;
    //        //        rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
    //        //        rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
    //        //    }

    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        unsigned int mask0 = 0xF0F0F0F0;
    //        unsigned int mask1 = 0x0F0F0F0F;
    //        unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
    //        unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

    //        //#pragma unroll
    //        //for(int i=0; i<4; i++){
    //        //    rhs_fragment_uint[i*2] = (rhs_fragment_transpose_uint[i*2] & mask1) | ((rhs_fragment_transpose_uint[i*2+1] & mask1) << 4);
    //        //    rhs_fragment_uint[i*2+1] = ((rhs_fragment_transpose_uint[i*2] & mask0) >> 4) | (rhs_fragment_transpose_uint[i*2+1] & mask0);
    //        //}
    //        
    //        rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
    //        rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
    //        rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
    //        rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
    //        rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
    //        rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
    //        rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
    //        rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);


    //        lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * 32];
    //        //lhs_fragment[0] = 0xFFFFFFFF;
    //        #pragma unroll
    //        for (int i = 0; i < 8; i ++){
    //            asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment[i])
    //            );
    //        }
    //    }


    //    __device__ __forceinline__ void TileMACResidue(){
    //        int lhs_fragment[1];
    //        int rhs_fragment[8];
    //        int rhs_fragment_transpose[8];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 136 + lane_id_ / 4;
    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);


    //        #pragma unroll
    //        for(int i=0; i<8; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }
    //        unsigned int mask0 = 0xF0F0F0F0;
    //        unsigned int mask1 = 0x0F0F0F0F;
    //        unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
    //        unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

    //        rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
    //        rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
    //        rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
    //        rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
    //        rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
    //        rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
    //        rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
    //        rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

    //        lhs_fragment[0] = lhs_tile_[lane_id_%32];
    //        #pragma unroll
    //        for (int i = 0; i < 8; i ++){
    //            asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment[i])
    //            );
    //        }
    //    }
    //};

    //// Tile_N=128 8bit v=2
    //template <int Tile_K>
    //struct wmmaComputeUtils_8b2v{

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils_8b2v(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int step){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        if(lane_id_ % 32 < 8)
    //            lhs_fragment[0] = lhs_tile_[lane_id_ % 8 + (step % 2) * 8];
    //        else
    //    	lhs_fragment[0] = 0;

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }


    //    __device__ __forceinline__ void TileMACResidue(){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        if(lane_id_ % 32 < 8)
    //            lhs_fragment[0] = lhs_tile_[lane_id_ % 8];
    //        else
    //    	lhs_fragment[0] = 0;

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }
    //};

    //// Tile_N=128 8bit v=4
    //template <int Tile_K>
    //struct wmmaComputeUtils_8b4v{

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils_8b4v(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int step){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        if(lane_id_ % 32 < 16)
    //            lhs_fragment[0] = lhs_tile_[lane_id_ % 16 + (step % 2) * 16];
    //        else
    //    	lhs_fragment[0] = 0;

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }


    //    __device__ __forceinline__ void TileMACResidue(){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        if(lane_id_ % 32 < 16)
    //            lhs_fragment[0] = lhs_tile_[lane_id_ % 16];
    //        else
    //    	lhs_fragment[0] = 0;

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }
    //};

    //template <int Tile_K>
    //struct wmmaComputeUtils_8b8v{

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils_8b8v(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int step){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * 32];

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }


    //    __device__ __forceinline__ void TileMACResidue(){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        lhs_fragment[0] = lhs_tile_[lane_id_ % 32];

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }
    //};

    // Tile_N=128 16-bit 8-bit 4 warps
    // TODO: Same as wmmaComputeUtils_8b?
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b8b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;
	    //int base_offset = chunk_id * 64 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 256;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
		//prefetching
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	        //lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

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
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;
	    //int base_offset = chunk_id * 64 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 256;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

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
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b8b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;
	    //int base_offset = chunk_id * 64 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 256;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }
            //prefetching
	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth];
	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth];
	    //lhs_fragment[0] = lhs_tile_[lane_id_ % 32];
	    //lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[4 + i]):
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
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[4 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;
	    //int base_offset = chunk_id * 64 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 256;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32];
	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[4 + i]):
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
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[4 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }
    };

    // Tile_N=64 16-bit 16-bit 4 warps
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[i%2*4 + i/2]), "+r"(output_fragment_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[i%2*4 + i/2]), "+r"(output_fragment_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }
    };

    // Tile_N=64 16-bit 16-bit 4 warps 8v
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[i%2*4 + i/2]), "+r"(output_fragment_0_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth];
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[i%2*4 + i/2]), "+r"(output_fragment_1_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[i%2*4 + i/2]), "+r"(output_fragment_0_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[i%2*4 + i/2]), "+r"(output_fragment_1_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

        }
    };

    // Tile_N=128 8-bit 4 warps
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_8b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_8b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    //int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;
	    int base_offset = chunk_id * 64 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 256;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
		//prefetching
	        //lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

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
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    //int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;
	    int base_offset = chunk_id * 64 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 256;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

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
        }
    };

    // Tile_N=128 4-bit 2 warps
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_4b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_4b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;

            //conflict-free opt
	    //int base_offset = chunk_id * 136 + lane_id_ / 4;
	    int base_offset = chunk_id * 128 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

	    //baseline with more bit-wise shift
	    unsigned char maskc = 0x0F;
            #pragma unroll
	    for(int j = 0; j < 8; j++)
	        for(int v = 0; v < 4; v++){
	    	    int intra_char_offset_0 = (j%2)*4;
	    	    int intra_char_offset_1 = ((j+1)%2)*4;
	            rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	            rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	        }

            //#pragma unroll
	    //for(int i=0; i<8; i++){
	    //    for(int j=0; j<4; j++){
	    //        *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	    //    }
	    //}

            //unsigned int mask0 = 0xF0F0F0F0;
            //unsigned int mask1 = 0x0F0F0F0F;
            //unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            //unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    //
	    //rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    //rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    //rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    //rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    //rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    //rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    //rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    //rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

            if(lane_id_ % 32 < ValuesBlockWidth)
		//prefetching
	        //lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;

            //conflict-free opt
	    //int base_offset = chunk_id * 136 + lane_id_ / 4;
	    int base_offset = chunk_id * 128 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);
	    //baseline with more bit-wise shift
	    unsigned char maskc = 0x0F;
            #pragma unroll
	    for(int j = 0; j < 8; j++)
	        for(int v = 0; v < 4; v++){
	    	    int intra_char_offset_0 = (j%2)*4;
	    	    int intra_char_offset_1 = ((j+1)%2)*4;
	            rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	            rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	        }

            //#pragma unroll
	    //for(int i=0; i<8; i++){
	    //    for(int j=0; j<4; j++){
	    //        *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	    //    }
	    //}
            //unsigned int mask0 = 0xF0F0F0F0;
            //unsigned int mask1 = 0x0F0F0F0F;
            //unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            //unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    //rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    //rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    //rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    //rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    //rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    //rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    //rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    //rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_8b4b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_8b4b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    // conflict-free
	    //int base_offset = chunk_id * 136 + lane_id_ / 4;
	    int base_offset = chunk_id * 128 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);
	    //baseline with more bit-wise shift
	    unsigned char maskc = 0x0F;
            #pragma unroll
	    for(int j = 0; j < 8; j++)
	        for(int v = 0; v < 4; v++){
	    	    int intra_char_offset_0 = (j%2)*4;
	    	    int intra_char_offset_1 = ((j+1)%2)*4;
	            rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	            rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	        }

            //#pragma unroll
	    //for(int i=0; i<8; i++){
	    //    for(int j=0; j<4; j++){
	    //        *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	    //    }
	    //}

            //unsigned int mask0 = 0xF0F0F0F0;
            //unsigned int mask1 = 0x0F0F0F0F;
            //unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            //unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    //
	    //rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    //rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    //rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    //rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    //rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    //rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    //rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    //rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    //ValuesBlockWidth = 64 for 8b 8v mma_dim_k=32
	    //prefetch
	    //lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth];
	    //lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth];
	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32];
	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];

            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }

            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    // conflict-free
	    //int base_offset = chunk_id * 136 + lane_id_ / 4;
	    int base_offset = chunk_id * 128 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);
	    //baseline with more bit-wise shift
	    unsigned char maskc = 0x0F;
            #pragma unroll
	    for(int j = 0; j < 8; j++)
	        for(int v = 0; v < 4; v++){
	    	    int intra_char_offset_0 = (j%2)*4;
	    	    int intra_char_offset_1 = ((j+1)%2)*4;
	            rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	            rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	        }

            //#pragma unroll
	    //for(int i=0; i<8; i++){
	    //    for(int j=0; j<4; j++){
	    //        *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	    //    }
	    //}
            //unsigned int mask0 = 0xF0F0F0F0;
            //unsigned int mask1 = 0x0F0F0F0F;
            //unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            //unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    //rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    //rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    //rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    //rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    //rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    //rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    //rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    //rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    //ValuesBlockWidth = 64 for 8b 8v mma_dim_k=32
	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32];
	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];

            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }

            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_12b4b2v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_12b4b2v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    
	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

            //if(lane_id_ % 32 < ValuesBlockWidth)
	    //    lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	    //else
	    //    lhs_fragment[0] = 0;
	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 32
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }
            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

            //if(lane_id_ % 32 < ValuesBlockWidth)
	    //    lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    //else
	    //    lhs_fragment[0] = 0;
	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 32
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_12b4b4v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_12b4b4v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    
	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }
            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }
        }
    };


    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_12b4b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int* output_fragment_2_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_12b4b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int* output_fragment_2,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1),
            output_fragment_2_(output_fragment_2){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[3];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    
	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 96
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 96
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[2] = lhs_tile_[lane_id_ % 32 + 64 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 96
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_2_[0 + i]), "+r"(output_fragment_2_[8 + i]):
                    "r"(lhs_fragment[2]),
                    "r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[3];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }
            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 96
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32]; // ValuesBlockWidth = 96
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[2] = lhs_tile_[lane_id_ % 32 + 64]; // ValuesBlockWidth = 96
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_2_[0 + i]), "+r"(output_fragment_2_[8 + i]):
                    "r"(lhs_fragment[2]),
                    "r"(rhs_fragment[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b4b4v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b4b4v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    
	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }
            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b4b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int* output_fragment_2_;
        int* output_fragment_3_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b4b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int* output_fragment_2,
            int* output_fragment_3,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1),
            output_fragment_2_(output_fragment_2),
            output_fragment_3_(output_fragment_3){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[4];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    
	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[2] = lhs_tile_[lane_id_ % 32 + 64 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_2_[0 + i]), "+r"(output_fragment_2_[8 + i]):
                    "r"(lhs_fragment[2]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[3] = lhs_tile_[lane_id_ % 32 + 96 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_3_[0 + i]), "+r"(output_fragment_3_[8 + i]):
                    "r"(lhs_fragment[3]),
                    "r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[4];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }
            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[8 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[2] = lhs_tile_[lane_id_ % 32 + 64]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_2_[0 + i]), "+r"(output_fragment_2_[8 + i]):
                    "r"(lhs_fragment[2]),
                    "r"(rhs_fragment[i])
                );
            }

	    lhs_fragment[3] = lhs_tile_[lane_id_ % 32 + 96]; // ValuesBlockWidth = 128
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_3_[0 + i]), "+r"(output_fragment_3_[8 + i]):
                    "r"(lhs_fragment[3]),
                    "r"(rhs_fragment[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_8b4b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_8b4b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    // conflict-free
	    //int base_offset = chunk_id * 136 + lane_id_ / 4;
	    int base_offset = chunk_id * 128 + lane_id_ / 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);
	    //baseline with more bit-wise shift
	    unsigned char maskc = 0x0F;
            #pragma unroll
	    for(int j = 0; j < 8; j++)
	        for(int v = 0; v < 4; v++){
	    	    int intra_char_offset_0 = (j%2)*4;
	    	    int intra_char_offset_1 = ((j+1)%2)*4;
	            rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	            rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	        }

            //#pragma unroll
	    //for(int i=0; i<8; i++){
	    //    for(int j=0; j<4; j++){
	    //        *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	    //    }
	    //}

            //unsigned int mask0 = 0xF0F0F0F0;
            //unsigned int mask1 = 0x0F0F0F0F;
            //unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            //unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);
	    //
	    //rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    //rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    //rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    //rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    //rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    //rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    //rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    //rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	        //lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[8];
            int rhs_fragment_transpose[8];
	    int chunk_id = lane_id_ % 4;
	    //conflict-free
	    //int base_offset = chunk_id * 136 + lane_id_ / 4;
	    int base_offset = chunk_id * 128 + lane_id_ / 4;
            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);
	    //baseline with more bit-wise shift
	    unsigned char maskc = 0x0F;
            #pragma unroll
	    for(int j = 0; j < 8; j++)
	        for(int v = 0; v < 4; v++){
	    	    int intra_char_offset_0 = (j%2)*4;
	    	    int intra_char_offset_1 = ((j+1)%2)*4;
	            rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	            rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	        }

            //#pragma unroll
	    //for(int i=0; i<8; i++){
	    //    for(int j=0; j<4; j++){
	    //        *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*4);
	    //    }
	    //}
            //unsigned int mask0 = 0xF0F0F0F0;
            //unsigned int mask1 = 0x0F0F0F0F;
            //unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            //unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

	    //rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    //rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    //rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    //rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    //rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    //rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    //rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    //rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;
            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                    //"r"(rhs_fragment[i])
                );
            }
        }
    };
    //// Tile_N=64 8bit v=4
    //template <int Tile_K>
    //struct wmmaComputeUtils_8b4v{

    //    // Shared memory buffers
    //    const int* lhs_tile_;
    //    const int* dense_tile_;
    //    // Register file fragment to accumulate results into
    //    int* output_fragment_;
    //    int lane_id_;

    //    // Constructor
    //    __device__ __forceinline__ wmmaComputeUtils_8b4v(
    //        const int* lhs_tile,
    //        const int* dense_tile,
    //        int* output_fragment,
    //        int lane_id):
    //        lhs_tile_(lhs_tile),
    //        lane_id_(lane_id),
    //        dense_tile_(dense_tile),
    //        output_fragment_(output_fragment){}
    //    
    //    // Compute
    //    __device__ __forceinline__ void TileMAC(int step){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + lane_id_ / 4;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        if(lane_id_ % 32 < 16)
    //            lhs_fragment[0] = lhs_tile_[lane_id_ % 16 + (step % 2) * 16];
    //        else
    //    	lhs_fragment[0] = 0;

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }


    //    __device__ __forceinline__ void TileMACResidue(){
    //        int lhs_fragment[1];
    //        int rhs_fragment[4];
    //        int rhs_fragment_transpose[4];
    //        int chunk_id = lane_id_ % 4;
    //        int base_offset = chunk_id * 72 + lane_id_ / 4;

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
    //        }

    //        unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
    //        unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

    //        #pragma unroll
    //        for(int i=0; i<4; i++){
    //            for(int j=0; j<4; j++){
    //                *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
    //            }
    //        }

    //        if(lane_id_ % 32 < 16)
    //            lhs_fragment[0] = lhs_tile_[lane_id_ % 16];
    //        else
    //    	lhs_fragment[0] = 0;

    //        #pragma unroll
    //        for (int i = 0; i < 4; i ++){
    //            asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
    //                "{%0, %1}, \t"
    //                "{%2}, \t"
    //                "{%3}, \t"
    //                "{%0, %1}; ":
    //                "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
    //                "r"(lhs_fragment[0]),
    //                "r"(rhs_fragment_transpose[i])
    //            );
    //        }
    //    }
    //};

    // Tile_N=128
    template <int Tile_K>
    struct wmmaComputeUtils4_4bit {

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils4_4bit(
            const int* lhs_tile,
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
            int rhs_fragment[16];
            int rhs_fragment_transpose[16];
	    int chunk_id = lane_id_ % 4;
	    int base_offset = chunk_id * 136 + lane_id_/4;
	    //int base_offset = chunk_id * 72 + lane_id_/4 + n_group_idx * 72 * 4;
	    //int base_offset = chunk_id * 64 + lane_id_/4 + n_group_idx * 64 * 4;

            #pragma unroll
	    for(int i=0; i<8; i++){
	        rhs_fragment[i*2] = *(dense_tile_ + base_offset + i*16); 
	        rhs_fragment[i*2+1] = *(dense_tile_ + base_offset + 8 + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

	    //baseline with more bit-wise shift
	    //unsigned char maskc = 0x0F;
            //#pragma unroll
	    //for(int j = 0; j < 8; j++)
	    //    for(int v = 0; v < 4; v++){
	    //	    int intra_char_offset_0 = (j%2)*4;
	    //	    int intra_char_offset_1 = ((j+1)%2)*4;
	    //        rhs_fragment_transpose_char[8*v+j/2] |= ((rhs_fragment_char[j*4+v] & maskc) << intra_char_offset_0);
	    //        rhs_fragment_transpose_char[8*v+8/2+j/2] |= ((rhs_fragment_char[j*4+v] & (maskc << 4)) >> intra_char_offset_1);
	    //    }

            #pragma unroll
	    for(int i=0; i<8; i++){
	        for(int j=0; j<8; j++){
	            *(rhs_fragment_transpose_char + j*8 + i) = *(rhs_fragment_char + j + i*8);
	        }
	    }
            unsigned int mask0 = 0xF0F0F0F0;
            unsigned int mask1 = 0x0F0F0F0F;
            unsigned int *rhs_fragment_uint = reinterpret_cast<unsigned int *>(rhs_fragment); 
            unsigned int *rhs_fragment_transpose_uint = reinterpret_cast<unsigned int *>(rhs_fragment_transpose);

            //#pragma unroll
	    //for(int i=0; i<4; i++){
	    //    rhs_fragment_uint[i*2] = (rhs_fragment_transpose_uint[i*2] & mask1) | ((rhs_fragment_transpose_uint[i*2+1] & mask1) << 4);
	    //    rhs_fragment_uint[i*2+1] = ((rhs_fragment_transpose_uint[i*2] & mask0) >> 4) | (rhs_fragment_transpose_uint[i*2+1] & mask0);
	    //}
	    
	    rhs_fragment_uint[0] = (rhs_fragment_transpose_uint[0] & mask1) | ((rhs_fragment_transpose_uint[1] & mask1) << 4);
	    rhs_fragment_uint[1] = ((rhs_fragment_transpose_uint[0] & mask0) >> 4) | (rhs_fragment_transpose_uint[1] & mask0);
	    rhs_fragment_uint[2] = (rhs_fragment_transpose_uint[2] & mask1) | ((rhs_fragment_transpose_uint[3] & mask1) << 4);
	    rhs_fragment_uint[3] = ((rhs_fragment_transpose_uint[2] & mask0) >> 4) | (rhs_fragment_transpose_uint[3] & mask0);
	    rhs_fragment_uint[4] = (rhs_fragment_transpose_uint[4] & mask1) | ((rhs_fragment_transpose_uint[5] & mask1) << 4);
	    rhs_fragment_uint[5] = ((rhs_fragment_transpose_uint[4] & mask0) >> 4) | (rhs_fragment_transpose_uint[5] & mask0);
	    rhs_fragment_uint[6] = (rhs_fragment_transpose_uint[6] & mask1) | ((rhs_fragment_transpose_uint[7] & mask1) << 4);
	    rhs_fragment_uint[7] = ((rhs_fragment_transpose_uint[6] & mask0) >> 4) | (rhs_fragment_transpose_uint[7] & mask0);

	    rhs_fragment_uint[8] = (rhs_fragment_transpose_uint[8] & mask1) | ((rhs_fragment_transpose_uint[9] & mask1) << 4);
	    rhs_fragment_uint[9] = ((rhs_fragment_transpose_uint[8] & mask0) >> 4) | (rhs_fragment_transpose_uint[9] & mask0);
	    rhs_fragment_uint[10] = (rhs_fragment_transpose_uint[10] & mask1) | ((rhs_fragment_transpose_uint[11] & mask1) << 4);
	    rhs_fragment_uint[11] = ((rhs_fragment_transpose_uint[10] & mask0) >> 4) | (rhs_fragment_transpose_uint[11] & mask0);
	    rhs_fragment_uint[12] = (rhs_fragment_transpose_uint[12] & mask1) | ((rhs_fragment_transpose_uint[13] & mask1) << 4);
	    rhs_fragment_uint[13] = ((rhs_fragment_transpose_uint[12] & mask0) >> 4) | (rhs_fragment_transpose_uint[13] & mask0);
	    rhs_fragment_uint[14] = (rhs_fragment_transpose_uint[14] & mask1) | ((rhs_fragment_transpose_uint[15] & mask1) << 4);
	    rhs_fragment_uint[15] = ((rhs_fragment_transpose_uint[14] & mask0) >> 4) | (rhs_fragment_transpose_uint[15] & mask0);

	    if(lane_id_ < 16){
	        lhs_fragment[0] = lhs_tile_[lane_id_+n_group_idx*16];
	    }
	    else{
	        lhs_fragment[0] = 0;
	    } 

            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[8 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i])
                );
            }

            #pragma unroll
            for (int i = 0; i < 8; i ++){
                asm("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[16 + i]), "+r"(output_fragment_[24 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment[i+8])
                );
            }
        }
    };

    // Compute Tile for k=4, 8-bit integer
    template <typename VecType, int Tile_K>
    struct wmmaComputeUtils4_8bit {
        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_K / 16 - 1;
        //
        // Member variables
        //

        // Shared memory buffers
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
    template <typename VecType, int Tile_K>
    struct wmmaComputeUtils4 {
        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_K / 4 - 1;
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
    template <typename VecType, int Tile_K>
    struct wmmaComputeUtils2 {
        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_K / 4 - 1;

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
