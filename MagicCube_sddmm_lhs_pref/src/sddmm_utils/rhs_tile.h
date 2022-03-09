namespace sddmm {
    template <int Tile_K, int MMA_Dim_K_int, int Blocks>
    struct wmma_rhs_4b{

        const int workset_;
        const int lane_id_;
        const int *rhs_matrix_base_;
        const int *column_indices_tile_;
        int *rhs_prefetch_;

        __device__ __forceinline__ wmma_rhs_4b(
            int workset, 
            int lane_id, 
            const int* __restrict__ rhs_matrix, 
            const int *column_indices_tile,
            int * rhs_prefetch):
            workset_(workset),
            lane_id_(lane_id),
            rhs_matrix_base_(rhs_matrix + column_indices_tile[lane_id/MMA_Dim_K_int]),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void Fetch(int step){
	    if(lane_id_/MMA_Dim_K_int < workset_){
                for(int i=0; i<Blocks; i++){
                    rhs_prefetch_[i] = __ldg(rhs_matrix_base_ + step * Tile_K + i * MMA_Dim_K_int + lane_id_ % MMA_Dim_K_int);
                }
	    }
        }

        //// Load the residual and compute the matrix product
        //__device__ __forceinline__ void ResidueLoad(int residue){
	//    printf("Residue not supported\n");
        //}
    };

    template <int Tile_K, int MMA_Dim_K_int, int Blocks>
    struct wmma_rhs_8b{

        const int workset_;
        const int lane_id_;
        const int *rhs_matrix_base_;
        const int *column_indices_tile_;
        int *rhs_prefetch_;

        __device__ __forceinline__ wmma_rhs_8b(
            int workset, 
            int lane_id, 
            const int* __restrict__ rhs_matrix, 
            const int *column_indices_tile,
            int * rhs_prefetch):
            workset_(workset),
            lane_id_(lane_id),
            rhs_matrix_base_(rhs_matrix + column_indices_tile[lane_id/MMA_Dim_K_int]),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void Fetch(int step){
	    if(lane_id_/MMA_Dim_K_int < workset_){
                for(int i=0; i<Blocks; i++){
                    rhs_prefetch_[i] = __ldg(rhs_matrix_base_ + step * Tile_K + i * MMA_Dim_K_int + lane_id_ % MMA_Dim_K_int);
                }
	    }
        }

        //// Load the residual and compute the matrix product
        //__device__ __forceinline__ void ResidueLoad(int residue){
	//    printf("Residue not supported\n");
        //}
    };

    template <int Tile_K, int MMA_Dim_K_int, int Blocks>
    struct wmma_rhs_16b{

        const int workset_;
        const int lane_id_;
        const int *rhs_matrix_base_;
        const int *column_indices_tile_;
        unsigned long long *rhs_prefetch_;

        __device__ __forceinline__ wmma_rhs_16b(
            int workset, 
            int lane_id, 
            const int* __restrict__ rhs_matrix, 
            const int *column_indices_tile,
            int * rhs_prefetch):
            workset_(workset),
            lane_id_(lane_id),
            rhs_matrix_base_(rhs_matrix + column_indices_tile[lane_id/MMA_Dim_K_int]),
            rhs_prefetch_(reinterpret_cast<unsigned long long *>(rhs_prefetch)){}
        

        __device__ __forceinline__ void Fetch(int step){
	    if(lane_id_/MMA_Dim_K_int < workset_){
                #pragma unroll
                for(int i=0; i<Blocks; i++){
                    //rhs_prefetch_[i * 2] = __ldg(rhs_matrix_base_ + step * Tile_K + i * MMA_Dim_K_int * 2 + lane_id_ % MMA_Dim_K_int * 2);
                    //rhs_prefetch_[i * 2 + 1] = __ldg(rhs_matrix_base_ + step * Tile_K + i * MMA_Dim_K_int * 2 + lane_id_ % MMA_Dim_K_int * 2 + 1);
                    rhs_prefetch_[i] = *(reinterpret_cast<const unsigned long long *>(rhs_matrix_base_ + step * Tile_K + i * MMA_Dim_K_int * 2 + lane_id_ % MMA_Dim_K_int * 2));
                }
	    }
        }

        //// Load the residual and compute the matrix product
        //__device__ __forceinline__ void ResidueLoad(int residue){
	//    printf("Residue not supported\n");
        //}
    };
}
