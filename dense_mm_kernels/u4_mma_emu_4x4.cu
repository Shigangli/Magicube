#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
//#include <helper_functions.h>

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 128

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 1

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 32
// one-byte "uint8_t" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW 0 // Updated for int4

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

__global__ void apmm_w4a4(const int4 *W, const int4 *X, int *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int wb, int xb) {
  // GEMM configuration.
  int K_TILES = K_GLOBAL / 128;

  int W_bit_offset = M_GLOBAL*K_GLOBAL/128;
  int X_bit_offset = N_GLOBAL*K_GLOBAL/128;
  int ROW_BIT = K_GLOBAL/128;

  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = block_pos / (N_GLOBAL/16) * 16;
    const unsigned int block_tile_j = block_pos % (N_GLOBAL/16) * 16;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_GLOBAL) {
      break;
    }

    typedef union {
      int4 vec;
      int a[4];
    } U4;

    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j = 0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const int4 *warp_ptr;
    if (warpId < 4) {
      warp_ptr = &W[block_tile_i * ROW_BIT + W_bit_offset*warpId];
    } else {
      warp_ptr = &X[block_tile_j * ROW_BIT + X_bit_offset*(warpId-4)];
    }

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Offset in shared memory from which the B matrix is stored.
      const size_t shmem_idx_b_off = 64; // TODO: This BLOCK_COL_TILES may be selected to improve performance. Maybe moved outside the for loop.

      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      int *shmem_ptr = (int*)shmem + warpId*16*4*(CHUNK_K+SKEW) + (laneId/4)*4*(CHUNK_K+SKEW) + laneId%4;

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      // int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * (K/128) +
      //                           (laneId / CHUNK_COPY_LINE_LANES) * (K_GLOBAL/128)) +
      //                  (laneId % CHUNK_COPY_LINE_LANES); // (K/128), since K=128 in bit. int4 is 128 bit.
      int *lane_ptr = (int*)warp_ptr + laneId/4*ROW_BIT*4 + laneId%4 + tile_k*4;
      
      *shmem_ptr = *lane_ptr;

      shmem_ptr += 8*4*(CHUNK_K+SKEW);
      lane_ptr += 8*ROW_BIT*4;
      *shmem_ptr = *lane_ptr;

      // U4 tmp_probe;
      // tmp_probe.vec = *lane_ptr;
      // printf("tmp_probe.a[0]: %d, tmp_probe.a[1]: %d, tmp_probe.a[2]: %d, tmp_probe.a[3]: %d\n", tmp_probe.a[0], tmp_probe.a[1], tmp_probe.a[2], tmp_probe.a[3]);

      __syncthreads();

      // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
      //   for(int i=0; i<64; i+=16) {
      //     printf("Load from GL. i: %d, val: %08x %08x %08x %08x \n", i, *((int*)&shmem[i][0]+0), *((int*)&shmem[i][0]+1), *((int*)&shmem[i][0]+2), *((int*)&shmem[i][0]+3));
      //   }
      // }

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

          // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
          //   for(int t=0; t<a[i].num_elements; t++) {
          //     printf("a[%d].x[%d]: %x\n", i, t, a[i].x[t]);
          //   }
          //   printf("shmem_idx_a: %d, k_step: %d\n", shmem_idx_a, k_step);
          // }


#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem[shmem_idx_b][k_step * (K/128)];

              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
            }
            // printf("ckpt4\n");

            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
          }
        }
      }
      __syncthreads();
    }
    
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem[0][0] +
                              (warpId / 2) * 64 * 16 +
                              (warpId % 2) * 32; // Will be used only when writing back D. May be moved outside the for loop. TODO.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = shmem_warp_tile_ptr + i * 64 * 8 + j * 8;
        wmma::store_matrix_sync(tile_ptr, c[i][j], 64, C_LAYOUT);
      }
    }

    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   for(int i=0; i<4; i++) {
    //     for(int j=0; j<4; j++) {
    //       printf("i: %d, j: %d, val: %d\n", i, j, *((int*)&shmem[0][0]+i*64*16+j*16));
    //     }
    //   }
    // }

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    // int *shmem_warp_stream_ptr = (int*)&shmem[0][0] + warpId * SHMEM_STRIDE * M; // Will be used only when writing back D. Maybe moved outside the for loop. TODO.
    size_t idx = threadIdx.x/16 * 64 + threadIdx.x%16;
    int *shmem_warp_stream_ptr = (int*)&shmem[0][0]+idx;
    int val = 0;
    int base_multiplier = 1;

#pragma unroll
    for (int i=0; i<4; i++) {
      int cur_multiplier = base_multiplier;
#pragma unroll
      for (int j=0; j<4; j++) {
        int tmp = *(shmem_warp_stream_ptr+16*j);
        val += (cur_multiplier*tmp);
        cur_multiplier *= 2;
      }
      base_multiplier *= 2;
      shmem_warp_stream_ptr += 16*64;
    }

    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   printf("val: %d\n", val);
    // }

    // This warp's pointer to the C matrix data to copy memory from to shared memory. 
    // TODO: May be moved outside the for loop.
    size_t gmem_idx = block_tile_i*N_GLOBAL + block_tile_j + (threadIdx.x/16)*N_GLOBAL + (threadIdx.x%16);

    // Now that shared memory contains all the D tiles, stream them to global memory.
    int *dst_gmem_warp_stream_ptr = &D[gmem_idx];
    
    *dst_gmem_warp_stream_ptr = val;

    __syncthreads();
  }
}

void init_matrices(int4 *W, int4 *X, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int W_BIT, int X_BIT){
  int *W_int = (int*) W;
  int *X_int = (int*) X;
  for(int b=0; b<W_BIT; b++) {
    for(int i = 0; i < M_GLOBAL; i++) {
      for(int j = 0; j < K_GLOBAL/32; j++) {
        // W_int[b*M_GLOBAL*K_GLOBAL/32 + i*K_GLOBAL/32+j] = 0xFFFFFFFF;
        // W_int[b*M_GLOBAL*K_GLOBAL/32 + i*K_GLOBAL/32+j] = i;
        W_int[b*M_GLOBAL*K_GLOBAL/32 + i*K_GLOBAL/32+j] = rand();
      }
    }
  }

  for(int b = 0; b<X_BIT; b++) {
    for(int i = 0; i < N_GLOBAL; i++) {
      for(int j = 0; j < K_GLOBAL/32; j++) {
        // X_int[b*N_GLOBAL*K_GLOBAL/32 + i*K_GLOBAL/32+j] = 0xFFFFFFFF;
        // X_int[i*K_GLOBAL/32+j] = i*M_GLOBAL + j;
        X_int[b*N_GLOBAL*K_GLOBAL/32 + i*K_GLOBAL/32+j] = rand();
      }
    }  
  }
}


int int_pow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp % 2)
           result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}

void compute_ref(int4 *W, int4 *X, int *ref_C, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int W_BIT, int X_BIT) {
  int *W_int = (int*) W;
  int *X_int = (int*) X;

  for (int m = 0; m < M_GLOBAL; m++) {
    for (int n = 0; n < N_GLOBAL; n++) {
      int tmp = 0;
      for(int xb=0; xb<X_BIT; xb++) {
        int X_Multiplier = int_pow(2,xb);
        for(int wb=0; wb<W_BIT; wb++) {
          int W_Multiplier = int_pow(2,wb);
          for(int k_tile=0; k_tile<K_GLOBAL/32; k_tile++) {
            int w_int = W_int[wb*M_GLOBAL*K_GLOBAL/32 + m*K_GLOBAL/32 + k_tile];
            int x_int = X_int[xb*N_GLOBAL*K_GLOBAL/32 + n*K_GLOBAL/32 + k_tile];
            for(int k=0; k<32; k++) {
              int mask = 1;
              int x_val = ((mask << k) & x_int) >> k;
              int w_val = ((mask << k) & w_int) >> k;
              tmp += X_Multiplier * W_Multiplier * x_val * w_val;
            }
          }
        }
      }
      ref_C[m*N_GLOBAL+n]= tmp;
    }
  }
}



void validate_results(int *C, int* ref_C, int M_, int N_) {
  // Assume K_GLOBAL and N_GLOBAL is a multiplier of 32.
  printf("Checking computed result for correctness: ");
  bool correct = true;
  double eps = 1.e-6;  // machine zero

  for(int i = 0; i < M_; i++) {
    for(int j = 0; j < N_; j++) {
      int idx = i*N_+j;
      double dst = fabs(C[idx] - ref_C[idx]);
      double abs = fabs(C[idx]) * fabs(ref_C[idx]);
      double ref_err = dst / abs;
      if (ref_err > eps) {
        // printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",, eps);
        printf("i: %d, j: %d, C: %d, ref_C: %d\n", i, j, C[idx], ref_C[idx]);
        // printf("non equal\n");
        correct = false;
      }
    }
  }
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}



// #define verify_output

int main(int argc, char **argv) {

  //int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  //checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

  int X_BIT = 4;
  int W_BIT = 4;

  //int M_GLOBAL = 64;
  //int M_GLOBAL = 1024;
  // int N_GLOBAL = 64;
  // int K_GLOBAL = 128;
  //for (int N_GLOBAL=128; N_GLOBAL<=2048; N_GLOBAL += 128 ) {
  for (int N_GLOBAL=512; N_GLOBAL<=8192; N_GLOBAL *= 2 ) {
    int K_GLOBAL = N_GLOBAL;
    int M_GLOBAL = N_GLOBAL;
  
    int4 *X = NULL;
    int4 *W = NULL;
    int *Output = NULL;
  
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&W), sizeof(int4) * M_GLOBAL * (K_GLOBAL/128)* W_BIT));
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&X), sizeof(int4) * N_GLOBAL * (K_GLOBAL/128) * X_BIT));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&Output), sizeof(int) * M_GLOBAL * N_GLOBAL));
    
    
#ifdef verify_output
    int4 *W_h = NULL;
    int4 *X_h = NULL;
    int *Output_h = NULL;
  
    W_h = (int4 *)malloc(sizeof(int4) * M_GLOBAL * (K_GLOBAL/128) * W_BIT);
    X_h = (int4 *)malloc(sizeof(int4) * N_GLOBAL * (K_GLOBAL/128) * X_BIT);
    Output_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
    printf("Preparing validation data for GPU...\n");
    init_matrices(W_h, X_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, W_BIT, X_BIT);
    checkCudaErrors(cudaMemcpy(W, W_h, sizeof(int4) * M_GLOBAL * (K_GLOBAL/128) * W_BIT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(X, X_h, sizeof(int4) * N_GLOBAL * (K_GLOBAL/128) * X_BIT, cudaMemcpyHostToDevice));
#endif
  
    int SHMEM_SZ = 65536;
    checkCudaErrors(cudaFuncSetAttribute(
      apmm_w4a4, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
  
    // Run ours NUM_PROFILES times and record time.
    float bmma_ms_avg = 0.0f;
    int NUM_PROFILES = 1000;
    for(int iter=0; iter<NUM_PROFILES; ++iter){
            float bmma_ms = 0.0f;
            cudaEvent_t bmma_start;
            cudaEvent_t bmma_end;
            cudaEventCreate(&bmma_start);
            cudaEventCreate(&bmma_end);
            cudaEventRecord(bmma_start);
            checkKernelErrors(
              (apmm_w4a4<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                    SHMEM_SZ>>>(W, X, Output, M_GLOBAL, N_GLOBAL, K_GLOBAL, W_BIT, X_BIT)));
                  cudaEventRecord(bmma_end);
            cudaEventSynchronize(bmma_end);
            cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
            cudaEventDestroy(bmma_start);
            cudaEventDestroy(bmma_end);
            bmma_ms_avg += bmma_ms;
    }
  
    bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;

    printf("V83, 64x64. M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d, X_BIT: %d, W_BIT: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL, X_BIT, W_BIT);
    printf("Time: %f ms\n", bmma_ms_avg);  
    printf("TOPS: %.2f\n", (((double)(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)/(bmma_ms_avg/1000.)) / 1e12);
  
  
#ifdef verify_output
  printf("Validating results...\n");
  checkCudaErrors(cudaMemcpy(Output_h, Output, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

  int *Output_ref = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

  /* Copmpute reference matrix on CPU */
  compute_ref(W_h, X_h, Output_ref, M_GLOBAL, N_GLOBAL, K_GLOBAL, W_BIT, X_BIT);

  /* validation results */
  validate_results(Output_h, Output_ref, M_GLOBAL, N_GLOBAL);
  free(W_h);
  free(X_h);
  free(Output_h);
  free(Output_ref);
#endif
  
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(W)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(X)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(Output)));
  
  }

  return EXIT_SUCCESS;
}
