namespace sddmm {
    template <int BlockWidth, int Tile_K_int, int WarpBlockWidth, int MMA_Dim_K_int>
    struct wmma_lhs_4b{
        const int dim_k_;
        const int lane_id_;
        const int *lhs_matrix_base_;
        int *lhs_prefetch_;
        int *lhs_tile_;

        __device__ __forceinline__ wmma_lhs_4b(
            const int dim_k,
            const int m_index,
            const int lane_id,
            const int *lhs_matrix,
            int *lhs_prefetch,
            int *lhs_tile):
            dim_k_(dim_k),
            lane_id_(lane_id),
            lhs_matrix_base_(lhs_matrix + dim_k*m_index),
            lhs_prefetch_(lhs_prefetch),
            lhs_tile_(lhs_tile){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
	    if(lane_id_ < BlockWidth){
		//lhs_tile_[lane_id_ + (step % 2) * BlockWidth] = lhs_prefetch_[0];
		lhs_tile_[lane_id_] = lhs_prefetch_[0];
	    }
        }

        __device__ __forceinline__ void Fetch(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_tile_[lane_id_] = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }

        __device__ __forceinline__ void Prefetch(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_prefetch_[0] = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }

        //// Load the residual and compute the matrix product
        //__device__ __forceinline__ void ResidueLoad(int residue){
	//    printf("Residue not supported\n");
        //}
    };

    template <int BlockWidth, int Tile_K_int, int WarpBlockWidth, int MMA_Dim_K_int>
    struct wmma_lhs_8b{
        const int dim_k_;
        const int lane_id_;
        const int *lhs_matrix_base_;
        int *lhs_prefetch_;
        int *lhs_tile_;

        __device__ __forceinline__ wmma_lhs_8b(
            const int dim_k,
            const int m_index,
            const int lane_id,
            const int *lhs_matrix,
            int *lhs_prefetch,
            int *lhs_tile):
            dim_k_(dim_k),
            lane_id_(lane_id),
            lhs_matrix_base_(lhs_matrix + dim_k*m_index),
            lhs_prefetch_(lhs_prefetch),
            lhs_tile_(lhs_tile){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
	    if(lane_id_ < BlockWidth){
		//lhs_tile_[lane_id_ + (step % 2) * BlockWidth] = lhs_prefetch_[0];
		lhs_tile_[lane_id_] = lhs_prefetch_[0];
	    }
        }

        __device__ __forceinline__ void Fetch(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_tile_[lane_id_] = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }

        __device__ __forceinline__ void Prefetch(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_prefetch_[0] = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }

        //// Load the residual and compute the matrix product
        //__device__ __forceinline__ void ResidueLoad(int residue){
	//    printf("Residue not supported\n");
        //}
    };


    template <int BlockWidth, int Tile_K_int, int WarpBlockWidth, int MMA_Dim_K_int>
    struct wmma_lhs_16b{
        const int dim_k_;
        const int lane_id_;
        const int *lhs_matrix_base_;
        int *lhs_prefetch_;
        int *lhs_tile_;

        __device__ __forceinline__ wmma_lhs_16b(
            const int dim_k,
            const int m_index,
            const int lane_id,
            const int *lhs_matrix,
            int *lhs_prefetch,
            int *lhs_tile):
            dim_k_(dim_k),
            lane_id_(lane_id),
            lhs_matrix_base_(lhs_matrix + dim_k*m_index),
            lhs_prefetch_(lhs_prefetch),
            lhs_tile_(lhs_tile){}
        
        __device__ __forceinline__ void LoadRowfromRegister(int step){
	    if(lane_id_ < BlockWidth){
		//lhs_tile_[lane_id_ + (step % 2) * BlockWidth] = lhs_prefetch_[0]; // padding
		lhs_tile_[lane_id_] = lhs_prefetch_[0]; // padding
	    }
        }

        __device__ __forceinline__ void Fetch(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_tile_[lane_id_]  = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }

        __device__ __forceinline__ void Prefetch(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_prefetch_[0] = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }
    };
}
