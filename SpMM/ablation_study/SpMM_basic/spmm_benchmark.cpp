#include <cuda_runtime.h>
#include <random>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include "include/bm_test_utils.h"
#include "include/cuda_spmm.cuh"
#include "include/wmma_spmm.cuh"
#include "include/cublas_gemm.cuh"
#include <fstream>
#include <string>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <iostream>

int power2n(int n){
    int exp=1;
    for(int i=0; i < n; i++)
        exp *= 2;
    return exp;
}

//template <typename TypeA>
//double compute_ref_integers(TypeA *A, int *B, int *ref_C, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int preA, int preB, int vec_length, int *row_offsets, int *col_indices, int m_vec) {
//    TypeA maskA = (TypeA)(power2n(preA)-1); //0b0000000011111111 for 8 bits
//    int maskB = power2n(preB)-1; //0b0000000011111111 for 8 bits
//
//    // Initialize the output matrix with 0
//    for(int i=0; i < M_GLOBAL * N_GLOBAL; i++){
//        ref_C[i] = 0;
//    }
//
//    double flops = 0;     
//    int b_tile = 32/preB;
//    printf("cpu: maskA %d, maskB %d, vec_length %d, b_tile %d\n", maskA, maskB, vec_length, b_tile); 
//    // traverse all the vector rows
//    for(int i=0; i < m_vec; i++){
//        // traverse all the nonzero columns in this row
//        for(int j=row_offsets[i]; j < row_offsets[i+1]; j++){
//            int col_idx = col_indices[j];
//	    TypeA A_vec_tile = A[j];
//            // traverse all the elements in the vector
//            for(int av=0; av < vec_length; av++){
//                int row_idx = i * vec_length + av;
//                int shift_a = av*preA;
//                int a_val = (int)(((maskA << shift_a) & A_vec_tile) >> shift_a);
//                for(int n=0; n < N_GLOBAL/b_tile; n++){
//		    int B_tile = B[col_idx * (N_GLOBAL/b_tile) + n];
//                    for(int bv=0; bv < b_tile; bv++){
//			int shift_b = bv*preB;
//                        int b_val = ((maskB << shift_b) & B_tile) >> shift_b;
//		        if(a_val>maskA || a_val<0 || b_val<0 || b_val>maskB)
//		            printf("cpu compute error");
//                        ref_C[row_idx*N_GLOBAL + n*b_tile + bv] += a_val*b_val;
//			//if(i == 1 && av == 1)
//		        //    printf("a_val %d, b_val %d, intermediate value %d\n", a_val, b_val, a_val*b_val);
//                        flops += 2.0;
//	            }
//                }
//            }
//        }
//    }
//    return flops;
//}

// vector type size larger than long long
template <typename TypeA>
double compute_ref_integers(TypeA *A, int *B, int *ref_C, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int preA, int preA_cut, int preB, int vec_length, int *row_offsets, int *col_indices, int m_vec, int scaleA) {
    //int maskA = (int)(power2n(preA)-1); //0b0000000011111111 for 8 bits
    //int maskB = power2n(preB)-1; //0b0000000011111111 for 8 bits
    //int maskA_cut = (int)(power2n(preA_cut)-1); //0b0000111111111111 for 12 bits
    
    TypeA maskA = (TypeA)(power2n(preA)-1); //0b0000000011111111 for 8 bits
    int maskB = power2n(preB)-1; //0b0000000011111111 for 8 bits
    TypeA maskA_cut = (TypeA)(power2n(preA_cut)-1); //0b0000111111111111 for 12 bits

    // Initialize the output matrix with 0
    for(int i=0; i < M_GLOBAL * N_GLOBAL; i++){
        ref_C[i] = 0;
    }

    TypeA *A_vec_tiles = new TypeA[scaleA];
    double flops = 0;     
    int b_tile = 32/preB;
    // printf("cpu: maskA %d, maskA_cut %d, maskB %d, vec_length %d, b_tile %d\n", maskA, maskA_cut, maskB, vec_length, b_tile); 
    // traverse all the vector rows
    for(int i=0; i < m_vec; i++){
        // traverse all the nonzero columns in this row
        for(int j=row_offsets[i]; j < row_offsets[i+1]; j++){
            int col_idx = col_indices[j];
	    for(int t=0; t<scaleA; t++)
	        A_vec_tiles[t] = A[j * scaleA + t];
            // traverse all the elements in the vector
	    for(int tt=0; tt<scaleA; tt++){
	        TypeA A_vec_tile = A_vec_tiles[tt];
                for(int av=0; av < vec_length/scaleA; av++){
                    int row_idx = i*vec_length + av + tt*vec_length/scaleA;
                    int shift_a = av*preA;
                    int a_val = (int)((((maskA << shift_a) & A_vec_tile) >> shift_a) & maskA_cut);
                    for(int n=0; n < N_GLOBAL/b_tile; n++){
	                int B_tile = B[col_idx * (N_GLOBAL/b_tile) + n];
                        for(int bv=0; bv < b_tile; bv++){
	            	    int shift_b = bv*preB;
                            int b_val = ((maskB << shift_b) & B_tile) >> shift_b;
	                    //if(a_val>maskA_cut || a_val<0 || b_val<0 || b_val>maskB)
	                    //    printf("cpu compute error: %d, %d, maskA_cut %d, maskB %d \n", a_val, b_val, maskA_cut, maskB);
                            ref_C[row_idx*N_GLOBAL + n*b_tile + bv] += a_val*b_val;
	            	    //if(i == 1 && av == 1)
	                    //    printf("a_val %d, b_val %d, intermediate value %d\n", a_val, b_val, a_val*b_val);
                            flops += 2.0;
	                }
                    }
                }
	    }
        }
    }
    return flops;
}

template <typename TypeA, typename TypeB, typename OutType, typename IndexType, typename DTypeVec, typename ITypeVec, cudaDataType_t DCuSPARSE>
void BmFN(std::string benchmark, int N, int vec_length, int kernel, bool sorted, bool func, int sparse, int preA, int preA_cut, int preB, int scaleA){

    // Open the benchmark file
    std::ifstream infile(benchmark, std::ifstream::in);
    std::string line;
    // get the Size of the benchmark
    std::getline(infile, line, ',');
    const int m_vec = std::stoi(line);
    const int dimM = m_vec * vec_length;
    std::getline(infile, line, ',');
    const int dimK = std::stoi(line);
    std::getline(infile, line, '\n');
    const int nonzeros_vec = std::stoi(line);
    const int nonzeros = nonzeros_vec * vec_length;
    const int dimN = N;
    int mma_k_dim = 16;

    if(preA_cut == 4 || preB == 4)
        mma_k_dim = 32;
    else
        mma_k_dim = 16;

    printf("preA %d, preA_cut %d, preB %d, vec_length %d \n", preA, preA_cut, preB, vec_length); 
    printf("m_vec %d, dimN %d, nonzeros_vec %d, dimk %d, mma_k_dim %d \n", m_vec, dimN, nonzeros_vec, dimK, mma_k_dim); 
    //std::cout << "preA: " << preA << "preA_cut: " << preA_cut << "preB: " << preB << "\n" ;
    //std::cout << "m_vec: " << m_vec << "n: " << dimN << "nonzeros_vec: " << nonzeros_vec << "k: " << dimK << "vec_length: " << vec_length << "mma_k_dim: " << mma_k_dim << "\n" ;

    // Create the A column indices

    std::default_random_engine generator;
    int d;
    cudaGetDevice(&d);
    printf("device = %d\n", d);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, d);

    printf("number of SMs: %d\n", deviceProp.multiProcessorCount);

    // SpMM
    if (sparse == 1){
        int *row_offsets = new int[m_vec + 1];
        for(int i = 0; i < m_vec + 1; i++){
            if (i == m_vec) std::getline(infile, line, '\n');
            else std::getline(infile, line, ' ');
            row_offsets[i] = std::stoi(line);
        }
        int *col_indices = new int[nonzeros_vec];
        IndexType *col_indices_sputnik = new IndexType[nonzeros_vec];
        for(int i = 0; i < nonzeros_vec; i++){
            std::getline(infile, line, ' ');
            col_indices[i] = std::stoi(line);
            col_indices_sputnik[i] = (IndexType)std::stoi(line);
        }

        int *aligned_row_offsets = new int[m_vec*2];
	int aligned_num_item = 0;
	//int warp_width = 32;
	//if(preA == 8 && preB == 8 && vec_length == 4)
	//    warp_width = 16;
	//else if(preA == 8 && preB == 8 && vec_length == 8)
	//    warp_width = 16;
	//else if(preA == 8 && preB == 4 && vec_length == 4)
	//    warp_width = 32;
        //else if(preA == 4 && preB == 4 && vec_length == 8)
        //    warp_width = 32;
	//else if(preA == 4 && preB == 4 && vec_length == 4)
        //    warp_width = 32;
        //    //warp_width = 64;

	aligned_row_offsets[0] = aligned_num_item;
	for(int i = 1; i < m_vec + 1; i++){
	    int num_item = row_offsets[i] - row_offsets[i-1];
            //ceiling
	    aligned_num_item += (num_item + mma_k_dim - 1) / mma_k_dim * mma_k_dim;
	    //aligned_num_item += (num_item + warp_width - 1) / warp_width * warp_width;
	    if(i != m_vec)
	        aligned_row_offsets[i*2] = aligned_num_item;
	    aligned_row_offsets[i*2-1] = aligned_row_offsets[i*2-2] + num_item;
	}

	std::cout << " nonzero_vec: " << nonzeros_vec << " aligned_ nonzero_vec: " << aligned_num_item  << "\n" ;
        int *aligned_col_indices = new int[aligned_num_item];
        int *aligned_col_indices_shuffle = new int[aligned_num_item];
	for(int i = 0; i < aligned_num_item; i++){
	    aligned_col_indices[i] = -1;
	    aligned_col_indices_shuffle[i] = -1;
	}

	for(int i = 1; i < m_vec + 1; i++){
	    int offset_begin = row_offsets[i-1];
	    int offset_end = row_offsets[i];
	    for(int j = offset_begin; j < offset_end; j++)
	        aligned_col_indices[aligned_row_offsets[(i-1)*2] + j - offset_begin] = col_indices[j];
	}

	for(int i = 0; i < aligned_num_item/8; i++){
	    for(int j = 0; j < 8; j++){
	        aligned_col_indices_shuffle[i*8 + (j%2)*4 + j/2] = aligned_col_indices[i*8 + j];
	    }
	}

	TypeA *values;
	TypeA *aligned_values;
	TypeA *aligned_values_transpose;
	TypeA *aligned_values_transpose_decompose;
	TypeB *rhs_matrix;
	assert(sizeof(TypeA) * 8 * scaleA / preA == vec_length);

        values = new TypeA[nonzeros * scaleA * preA / (sizeof(TypeA)*8)];
        rhs_matrix = new TypeB[dimK * dimN * preB / (sizeof(TypeB)*8)];

        MakeDenseMatrix<TypeA>(1, nonzeros * scaleA * preA / (sizeof(TypeA)*8), values, generator);
        MakeDenseMatrix<TypeB>(dimK, dimN * preB / (sizeof(TypeB)*8), rhs_matrix, generator);

        aligned_values = new TypeA[aligned_num_item * scaleA];
        aligned_values_transpose = new TypeA[aligned_num_item * scaleA];
        aligned_values_transpose_decompose = new TypeA[aligned_num_item * scaleA];
	for(int i = 0; i < aligned_num_item * scaleA; i++){
	    aligned_values[i] = 0;
	    aligned_values_transpose[i] = 0;
	    aligned_values_transpose_decompose[i] = 0;
	}

	for(int i = 1; i < m_vec + 1; i++){
	    int offset_begin = row_offsets[i-1] * scaleA;
	    int offset_end = row_offsets[i] * scaleA;
	    for(int j = offset_begin; j < offset_end; j++)
	        aligned_values[aligned_row_offsets[(i-1)*2] * scaleA + j - offset_begin] = values[j];
	}

	// mma_k_dim-wise transpose for 8-bit int
	unsigned char * aligned_values_char = reinterpret_cast<unsigned char *>(aligned_values);
	unsigned char * aligned_values_transpose_char = reinterpret_cast<unsigned char *>(aligned_values_transpose);
	unsigned char * aligned_values_transpose_decompose_char = reinterpret_cast<unsigned char *>(aligned_values_transpose_decompose);

	// mma_k_dim-wise transpose for 12-bit int
	unsigned short * aligned_values_short = reinterpret_cast<unsigned short *>(aligned_values);
	unsigned short * aligned_values_transpose_short = reinterpret_cast<unsigned short *>(aligned_values_transpose);
	unsigned short * aligned_values_transpose_decompose_short = reinterpret_cast<unsigned short *>(aligned_values_transpose_decompose);

        // for 8-bit int
	if(preA_cut == 8){
	    for(int i = 0; i < aligned_num_item*vec_length; i+=(mma_k_dim*vec_length))
	        for(int j = 0; j < mma_k_dim; j++)
	            for(int v = 0; v < vec_length; v++)
	                aligned_values_transpose_char[i+v*mma_k_dim+j] = aligned_values_char[i+j*vec_length+v];

	    //for mixed precision
	    if(mma_k_dim == 32){
		unsigned char mask = 15;
	        for(int i = 0; i < aligned_num_item*vec_length; i+=(mma_k_dim*vec_length))
	            for(int j = 0; j < mma_k_dim*vec_length; j++){
			int intra_char_offset_0 = (j%2)*4;
			int intra_char_offset_1 = ((j+1)%2)*4;
			aligned_values_transpose_decompose_char[i+j/2] |= ((aligned_values_transpose_char[i+j] & mask) << intra_char_offset_0);
			aligned_values_transpose_decompose_char[i+mma_k_dim*vec_length/2+j/2] |= ((aligned_values_transpose_char[i+j] & (mask << 4)) >> intra_char_offset_1);
		    }
	    }
	}
	else if(preA_cut == 12 && mma_k_dim == 32){
	    for(int i = 0; i < aligned_num_item*vec_length; i+=(mma_k_dim*vec_length))
	        for(int j = 0; j < mma_k_dim; j++)
	            for(int v = 0; v < vec_length; v++)
	                aligned_values_transpose_short[i+v*mma_k_dim+j] = aligned_values_short[i+j*vec_length+v];

            unsigned short mask = 15;
            for(int i = 0; i < aligned_num_item*vec_length; i+=(mma_k_dim*vec_length))
                for(int j = 0; j < mma_k_dim*vec_length; j++){
		    if(j%4 == 0){
            	        aligned_values_transpose_decompose_short[i+j/4] |= (aligned_values_transpose_short[i+j] & mask);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 4)) >> 4);
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 8)) >> 8);
		    }
		    if(j%4 == 1){
            	        aligned_values_transpose_decompose_short[i+j/4] |= ((aligned_values_transpose_short[i+j] & mask) << 4);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= (aligned_values_transpose_short[i+j] & (mask << 4));
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 8)) >> 4);
		    }
		    if(j%4 == 2){
            	        aligned_values_transpose_decompose_short[i+j/4] |= ((aligned_values_transpose_short[i+j] & mask) << 8);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 4)) << 4);
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= (aligned_values_transpose_short[i+j] & (mask << 8));
		    }
		    if(j%4 == 3){
            	        aligned_values_transpose_decompose_short[i+j/4] |= ((aligned_values_transpose_short[i+j] & mask) << 12);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 4)) << 8);
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 8)) << 4);
		    }
                }
	}
	else if(preA_cut == 12 && mma_k_dim == 16){
	    unsigned char mask = 15;
	    for(int i = 0; i < aligned_num_item*vec_length*2; i+=(mma_k_dim*vec_length*2))
	        for(int j = 0; j < mma_k_dim; j++)
	            for(int v = 0; v < vec_length*2; v+=2){
	                aligned_values_transpose_decompose_char[i+j+(v/2)*mma_k_dim] = aligned_values_char[i+j*vec_length*2+v];
	                aligned_values_transpose_decompose_char[i+mma_k_dim*vec_length+j+(v/2)*mma_k_dim] = aligned_values_char[i+j*vec_length*2+v+1] & mask;
		    }
	}
	else if(preA_cut == 16 && mma_k_dim == 32){
	    for(int i = 0; i < aligned_num_item*vec_length; i+=(mma_k_dim*vec_length))
	        for(int j = 0; j < mma_k_dim; j++)
	            for(int v = 0; v < vec_length; v++)
	                aligned_values_transpose_short[i+v*mma_k_dim+j] = aligned_values_short[i+j*vec_length+v];

            unsigned short mask = 15;
            for(int i = 0; i < aligned_num_item*vec_length; i+=(mma_k_dim*vec_length))
                for(int j = 0; j < mma_k_dim*vec_length; j++){
		    if(j%4 == 0){
            	        aligned_values_transpose_decompose_short[i+j/4] |= (aligned_values_transpose_short[i+j] & mask);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 4)) >> 4);
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 8)) >> 8);
            	        aligned_values_transpose_decompose_short[i+3*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 12)) >> 12);
		    }
		    if(j%4 == 1){
            	        aligned_values_transpose_decompose_short[i+j/4] |= ((aligned_values_transpose_short[i+j] & mask) << 4);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= (aligned_values_transpose_short[i+j] & (mask << 4));
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 8)) >> 4);
            	        aligned_values_transpose_decompose_short[i+3*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 12)) >> 8);
		    }
		    if(j%4 == 2){
            	        aligned_values_transpose_decompose_short[i+j/4] |= ((aligned_values_transpose_short[i+j] & mask) << 8);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 4)) << 4);
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= (aligned_values_transpose_short[i+j] & (mask << 8));
            	        aligned_values_transpose_decompose_short[i+3*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 12)) >> 4);
		    }
		    if(j%4 == 3){
            	        aligned_values_transpose_decompose_short[i+j/4] |= ((aligned_values_transpose_short[i+j] & mask) << 12);
            	        aligned_values_transpose_decompose_short[i+mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 4)) << 8);
            	        aligned_values_transpose_decompose_short[i+2*mma_k_dim*vec_length/4+j/4] |= ((aligned_values_transpose_short[i+j] & (mask << 8)) << 4);
            	        aligned_values_transpose_decompose_short[i+3*mma_k_dim*vec_length/4+j/4] |= (aligned_values_transpose_short[i+j] & (mask << 12));
		    }
                }
	}
	else if(preA_cut == 16 && mma_k_dim == 16){
	    for(int i = 0; i < aligned_num_item*vec_length*2; i+=(mma_k_dim*vec_length*2))
	        for(int j = 0; j < mma_k_dim; j++)
	            for(int v = 0; v < vec_length*2; v+=2){
	                aligned_values_transpose_decompose_char[i+j+(v/2)*mma_k_dim] = aligned_values_char[i+j*vec_length*2+v];
	                aligned_values_transpose_decompose_char[i+mma_k_dim*vec_length+j+(v/2)*mma_k_dim] = aligned_values_char[i+j*vec_length*2+v+1];
		    }
	}
	else if(preA_cut == 4){ // for 4-bit int
	    unsigned char mask = 15; // 0b00001111
	    for(int i = 0; i < aligned_num_item*(vec_length/2); i+=(mma_k_dim*(vec_length/2)))
	        for(int j = 0; j < mma_k_dim; j++)
	            for(int v = 0; v < vec_length/2; v++){
			int intra_char_offset_0 = (j%2)*4;
			int intra_char_offset_1 = ((j+1)%2)*4;
	                aligned_values_transpose_char[i+mma_k_dim*v+j/2] |= ((aligned_values_char[i+j*(vec_length/2)+v] & mask) << intra_char_offset_0);
	                aligned_values_transpose_char[i+mma_k_dim*v+mma_k_dim/2+j/2] |= ((aligned_values_char[i+j*(vec_length/2)+v] & (mask << 4)) >> intra_char_offset_1);
			//if((i<8) && (j<8))
			//    printf("%x, %x, %x\n", aligned_values_char[i+j*(vec_length/2)+v], aligned_values_transpose_char[i+mma_k_dim*v+j/2], aligned_values_transpose_char[i+mma_k_dim*v+mma_k_dim/2+j/2]);
		    }
	}
	//}

        // Allocate the host output
        int *output_value_host = new int[dimM * dimN];
        double flops = 0;

        if(func){
            flops = compute_ref_integers<TypeA>(values, rhs_matrix, output_value_host, dimM, dimK, dimN, preA, preA_cut, preB, vec_length, row_offsets, col_indices, m_vec, scaleA);
            //if(mixed == 2){
            //    flops = compute_ref_integers<TypeA>(values, rhs_matrix, output_value_host, m, n, k, 8, 8, 4, row_offsets, col_indices, m_vec);
            //}
	    //else{
            //    // Initialize the output matrix with 0
            //    for (int i=0; i < m * k; i++){
            //        output_value_host[i] = 0.0f;
            //    }
            //    
            //    // traverse all the vector rows
            //    for (int i=0; i < m_vec; i++){
            //        // traverse all the nonzero columns in this row
            //        for (int j=row_offsets[i]; j < row_offsets[i+1]; j++){
            //            int col_idx = col_indices[j];
            //            // traverse all the elements in the vector
            //            for (int v=0; v < vec_length; v++){
            //                int row_idx = i * vec_length + v;
            //                for (int l=0; l < k; l++){
            //                    output_value_host[row_idx * k + l] += (float)values[j * vec_length + v] * (float)rhs_matrix[col_idx * k + l];
            //                    flops += 2.0;
            //                }
            //            }
            //        }
            //    }
	    //}
        }// end if func

	flops = flops/1000.0/1000.0/1000.0;
        std::cout << "total Gflops: " << flops << "\n";

        int *row_indices = new int[m_vec];
        if(sorted){
            //printf("Sort CSR based on row length\n");
            SortedRowSwizzle(m_vec, row_offsets, row_indices);
        }
        else{
            //printf("Process the rows in order\n");
            IdentityRowSwizzle(m_vec, row_indices);
        }
        
        // Device
        int *d_row_offsets, *d_col_indices, *d_row_indices;
        IndexType *d_col_indices_sputnik;
        int *d_values; 
	TypeB *d_rhs_matrix;
        OutType *d_output_value;
	int *aligned_values_transpose_decompose_int = reinterpret_cast<int *>(aligned_values_transpose_decompose);
	int *aligned_values_transpose_int = reinterpret_cast<int *>(aligned_values_transpose);

        checkCuda(cudaMalloc(&d_row_offsets, (m_vec*2) * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices, aligned_num_item * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices_sputnik, nonzeros_vec * sizeof(IndexType)));
        checkCuda(cudaMalloc(&d_row_indices, m_vec * sizeof(int)));

	
        checkCuda(cudaMalloc(&d_values, aligned_num_item * sizeof(TypeA) * scaleA));
        checkCuda(cudaMalloc(&d_rhs_matrix, dimK * dimN * preB / 8));
        checkCuda(cudaMalloc(&d_output_value, (dimM * dimN) * sizeof(OutType)));

        checkCuda(cudaMemcpy(d_row_offsets, aligned_row_offsets , (m_vec*2) * sizeof(int), cudaMemcpyHostToDevice));
	if(mma_k_dim == 16){
            checkCuda(cudaMemcpy(d_col_indices, aligned_col_indices, aligned_num_item * sizeof(int), cudaMemcpyHostToDevice));
	}
	else if(mma_k_dim == 32){
            checkCuda(cudaMemcpy(d_col_indices, aligned_col_indices_shuffle, aligned_num_item * sizeof(int), cudaMemcpyHostToDevice));
	}
        
	if(preA_cut > preB || (preA_cut == 16 && preB == 16))
            checkCuda(cudaMemcpy(d_values, aligned_values_transpose_decompose_int, aligned_num_item * sizeof(TypeA) * scaleA, cudaMemcpyHostToDevice));
	else
            checkCuda(cudaMemcpy(d_values, aligned_values_transpose_int, aligned_num_item * sizeof(TypeA) * scaleA, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, dimK * dimN * preB / 8, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_row_indices, row_indices, m_vec * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices_sputnik, col_indices_sputnik, nonzeros_vec * sizeof(IndexType), cudaMemcpyHostToDevice));
        
        cudaProfilerStart();
	float spmm_ms_avg = 0.0f;
	int NUM_PROFILES = 1024;
        if((kernel == 0) && (preA_cut == 4) && (preB == 4)){
            printf("Using WMMA \n");
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_4b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 8) && (preB == 4)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_8b4b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 12) && (preB == 4)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_12b4b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 16) && (preB == 4)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_16b4b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 8) && (preB == 8)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                //spmm::wmmaSpmm_8b4v(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
                spmm::wmmaSpmm_8b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 12) && (preB == 8)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_12b8b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 16) && (preB == 8)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_16b8b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if((kernel == 0) && (preA_cut == 16) && (preB == 16)){
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm_16b(m_vec, vec_length, dimN, dimK, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
	else if(kernel == 0){
            //printf("Using WMMA \n");
	    //for(int iter=0; iter<NUM_PROFILES; ++iter){
	    //    float spmm_ms = 0.0f;
	    //    cudaEvent_t spmm_start;
	    //    cudaEvent_t spmm_end;
	    //    cudaEventCreate(&spmm_start);
	    //    cudaEventCreate(&spmm_end);
	    //    cudaEventRecord(spmm_start);
            //    spmm::wmmaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
	    //    cudaEventRecord(spmm_end);
	    //    cudaEventSynchronize(spmm_end);
	    //    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
            //    cudaEventDestroy(spmm_start);
            //    cudaEventDestroy(spmm_end);
            //    spmm_ms_avg += spmm_ms;
	    //}
            //spmm_ms_avg = spmm_ms_avg/(float)NUM_PROFILES/1000.0;
            //std::cout << "performance GFLOP/s: " << flops/spmm_ms_avg << "\n";
            printf("Unsupported Kernel \n");
        }
        else{
            printf("Unsupported Kernel \n");
        }
        spmm_ms_avg = spmm_ms_avg/(float)NUM_PROFILES/1000.0;
        std::cout << "performance TOP/s: " << flops/spmm_ms_avg/1000.0 << "\n";
        cudaProfilerStop();

        //else if (kernel == 1){
        //    printf("Using CUDA \n");
        //    spmm::cudaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_values, d_rhs_matrix, d_output_value);
        //}
        //else if (kernel == 2){
        //    printf("Using Sputnik \n");
        //    DTypeVec* d_values_vec = reinterpret_cast<DTypeVec *>(d_values);
        //    DTypeVec* d_rhs_matrix_vec = reinterpret_cast<DTypeVec *>(d_rhs_matrix);
        //    DTypeVec* d_output_value_vec = reinterpret_cast<DTypeVec *>(d_output_value);
        //    ITypeVec* d_col_indices_sputnik_vec = reinterpret_cast<ITypeVec *>(d_col_indices_sputnik);
        //    sputnik::CudaSpmm(m, n, k, nonzeros, d_row_indices, d_values_vec, d_row_offsets, d_col_indices_sputnik_vec, d_rhs_matrix_vec, d_output_value_vec, 0);
        //}
        //else if (kernel == 3){
        //    printf("Using CuSPARSE \n");
        //    cusparseHandle_t handle;
        //    cusparseDnMatDescr_t rhs_dense, output_dense;
        //    cusparseSpMatDescr_t lhs_sparse;

        //    cusparseCreate(&handle);

        //    // create lhs sparse matrix
        //    cusparseCreateCsr(
        //        &lhs_sparse, m, n, nonzeros_vec, d_row_offsets, d_col_indices, d_values,
        //        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, DCuSPARSE
        //    );
        //    
        //    // create rhs dense matrix
        //    cusparseCreateDnMat(
        //        &rhs_dense, n, k, k, d_rhs_matrix, DCuSPARSE, CUSPARSE_ORDER_ROW
        //    );

        //    // create output dense matrix
        //    cusparseCreateDnMat(
        //        &output_dense, m, k, k, d_output_value, DCuSPARSE, CUSPARSE_ORDER_ROW
        //    );

        //    InType alpha = 1.0;
        //    InType beta  = 0.0;
        //    size_t buffer_size;
        //    void* dBuffer = NULL;

        //    // get buffer
        //    cusparseSpMM_bufferSize(
        //        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //        &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_CSR_ALG2, &buffer_size
        //    );

        //    checkCuda(cudaMalloc(&dBuffer, buffer_size));
        //    
        //    /*
        //    // preprocess to get additional speedup
        //    cusparseSpMM_preprocess(
        //        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //        &alpha, lhs_sparse, rhs_dense, &beta, output_dense, CUDA_R_16F, CUSPARSE_SPMM_CSR_ALG2,
        //        dBuffer
        //    );
        //    */

        //    // execute SpMM
        //    cusparseSpMM(
        //        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //        &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_CSR_ALG2,
        //        dBuffer
        //    );

        //    checkCuda(cudaFree(dBuffer));
        //    cusparseDestroyDnMat(rhs_dense);
        //    cusparseDestroyDnMat(output_dense);
        //    cusparseDestroySpMat(lhs_sparse);
        //    cusparseDestroy(handle);
        //}


        if (func){
            OutType *output_value_cuda = new OutType[dimM * dimN];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, dimM * dimN * sizeof(OutType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            int counter = 0;
            for (int j=0; j < dimM * dimN; j++){
		if (output_value_cuda[j] > 0) counter++;
                if ((output_value_cuda[j] - output_value_host[j]) != 0){
                    errors ++;
                }
            }
            if (errors > 0) {
                //printf( "SPMM does not agree with SEQUENTIAL! %d errors!\n",errors);
            }else {
                printf("Results verification: PASS\n");
            }
	    printf("counter = %d\n", counter);
            delete output_value_cuda;
        }


        // Free the memory
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_row_indices);
        cudaFree(d_col_indices_sputnik);
        cudaFree(d_values);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);

        delete row_offsets;
        delete col_indices;
        delete col_indices_sputnik;
        delete row_indices;
        delete values;
        delete rhs_matrix;
        delete output_value_host;
    }
}

int main(int argc, char **argv){
    // Helper function
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)){
        printf("This script does a A_mxk * B_kxn = C_mxn matrix multiplication.\n");
        printf("The A_mxn can be a sparse matrix in CSR format loaded from the benchmark [bm], or a row-major dense matrix.\n");
        printf("The B_kxn and C_mxn are row-major dense matrices.\n");
        printf("\n");
        printf("usage: ./spmm_benchmark [bm] [n] [v] [kernel] [sort] [function] [sparse] [preA] [preB]\n");
        printf("arguments\n");
        printf("bm      :   path to the sparse matrix benchmark.\n");
        printf("            e.g.: /raid/datasets/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx\n");
        printf("n       :   the length of dimension n.\n");
        printf("v       :   the vector length of the column vector sparsity, can be {1, 2, 4, 8}. \n");
        printf("kernel  :   kernel = 0 & v=2, 4, 8,    the wmmaSpMM is used. \n");
        printf("            kernel = 1 & v=1, 2, 4, 8, the cudaSpMM is used. \n");
        printf("            kernel = 2 & v=1, the sputnik is used. \n");
        printf("            kernel = 3 & v=1, the cusparse is used. \n");
        printf("sort    :   sort = 1, the rows are sorted to balance the workload; \n");
        printf("            sort = 0, the rows are processed in order; \n");
        printf("function:   function = 1, the result of the kernel will be verified.\n");
        printf("            function = 0, the result verification is skipped\n");
        printf("sparse  :   sparse = 0, the dense version is executed as a baseline;\n");
        printf("            sparse = 1, the SpMM is executed;\n");
        printf("        :   sparse = 2, the Blocked Ell based SpMM is executed");
        printf("preA   :   preA = 32, use single precision; \n");
        printf("           preA = 16, use half precision; \n");
        printf("           preA = 12, use 12-bit int precision; \n");
        printf("           preA = 8, use 8-bit int precision; \n");
        printf("           preA = 4, use 4-bit int precision; \n");
        printf("preB   :   preB = 32, use single precision; \n");
        printf("           preB = 16, use half precision; \n");
        printf("           preB = 12, use 12-bit int precision; \n");
        printf("           preB = 8, use 8-bit int precision; \n");
        printf("           preB = 4, use 4-bit int precision; \n");
    }
    // Run the benchmark
    else{
        std::string benchmark(argv[1]);
        int dimN = std::atoi(argv[2]);
        int vec_length = std::atoi(argv[3]);
        int kernel = std::atoi(argv[4]);
        int sorted = std::atoi(argv[5]);
        int func = std::atoi(argv[6]);
        int sparse = std::atoi(argv[7]);
        int preA = std::atoi(argv[8]);
        int preB = std::atoi(argv[9]);
        std::cout << "Sparse matrix: " << benchmark << "\n" ;

	if ((preA == 4) && (preB == 4) && (vec_length == 8)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 4) && (preB == 4) && (vec_length == 4)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 4) && (preB == 4) && (vec_length == 2)) BmFN<char, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 8) && (preB == 4) && (vec_length == 8)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 8) && (preB == 4) && (vec_length == 4)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 8) && (preB == 4) && (vec_length == 2)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 12) && (preB == 4) && (vec_length == 8)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, 16, preA, preB, 2);
	else if ((preA == 12) && (preB == 4) && (vec_length == 4)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, 16, preA, preB, 1);
	else if ((preA == 12) && (preB == 4) && (vec_length == 2)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, 16, preA, preB, 1);
	else if ((preA == 16) && (preB == 4) && (vec_length == 8)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 2);
	else if ((preA == 16) && (preB == 4) && (vec_length == 4)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 16) && (preB == 4) && (vec_length == 2)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 8) && (preB == 8) && (vec_length == 8)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 8) && (preB == 8) && (vec_length == 4)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 8) && (preB == 8) && (vec_length == 2)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 12) && (preB == 8) && (vec_length == 8)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, 16, preA, preB, 2);
	else if ((preA == 12) && (preB == 8) && (vec_length == 4)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, 16, preA, preB, 1);
	else if ((preA == 12) && (preB == 8) && (vec_length == 2)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, 16, preA, preB, 1);
	else if ((preA == 16) && (preB == 8) && (vec_length == 8)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 2);
	else if ((preA == 16) && (preB == 8) && (vec_length == 4)) BmFN<long long, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	else if ((preA == 16) && (preB == 8) && (vec_length == 2)) BmFN<int, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 1);
	//else if ((preA == 16) && (preB == 8) && (vec_length == 8)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 8);
	//else if ((preA == 16) && (preB == 8) && (vec_length == 4)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 4);
	//else if ((preA == 16) && (preB == 8) && (vec_length == 2)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 2);
	else if ((preA == 16) && (preB == 16) && (vec_length == 8)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 8);
	else if ((preA == 16) && (preB == 16) && (vec_length == 4)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 4);
	else if ((preA == 16) && (preB == 16) && (vec_length == 2)) BmFN<short, int, int, short, half2, short2, CUDA_R_16F>(benchmark, dimN, vec_length, kernel, sorted, func, sparse, preA, preA, preB, 2);
	else printf("Unsupported precision and vec_length!\n");
    }
    
}
