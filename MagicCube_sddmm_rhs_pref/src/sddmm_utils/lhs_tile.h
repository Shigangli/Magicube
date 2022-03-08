namespace sddmm {
    template <int BlockWidth, int Tile_K_int, int WarpBlockWidth, int MMA_Dim_K_int>
    struct wmma_lhs_4b{
        const int dim_k_;
        const int lane_id_;
        const int *lhs_matrix_base_;
        int *lhs_tile_;

        __device__ __forceinline__ wmma_lhs_4b(
            const int dim_k,
            const int m_index,
            const int lane_id,
            const int *lhs_matrix,
            int *lhs_tile):
            dim_k_(dim_k),
            lane_id_(lane_id),
            lhs_matrix_base_(lhs_matrix + dim_k*m_index),
            lhs_tile_(lhs_tile){}
        
        __device__ __forceinline__ void Load(int step){
	    const int offset_int = Tile_K_int * step;
	    if(lane_id_ < BlockWidth){
		lhs_tile_[lane_id_] = __ldg(lhs_matrix_base_ + offset_int + lane_id_ / WarpBlockWidth * MMA_Dim_K_int + lane_id_ % WarpBlockWidth / MMA_Dim_K_int * dim_k_ + lane_id_ % MMA_Dim_K_int);
	    }
        }


        //// Load the residual and compute the matrix product
        //__device__ __forceinline__ void ResidueLoad(int residue){
	//    printf("Residue not supported\n");
        //}
    };
}
