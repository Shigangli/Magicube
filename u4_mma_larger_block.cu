#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.


#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 32

// GEMM configuration.

//#define M_TILES 32
//#define N_TILES 32
//#define K_TILES 32

//#define M_TILES 64
//#define N_TILES 64
//#define K_TILES 64

//#define M_TILES 128
//#define N_TILES 128
//#define K_TILES 128

//#define M_TILES 256
//#define N_TILES 256
//#define K_TILES 256

//#define M_TILES 512
//#define N_TILES 512
//#define K_TILES 512

//#define M_TILES 1024
//#define N_TILES 1024
//#define K_TILES 1024

//#define M_TILES 2048
//#define N_TILES 2048
//#define K_TILES 2048

//#define M_GLOBAL (M * M_TILES)
//#define N_GLOBAL (N * N_TILES)
//#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 8
#else
#define CHUNK_K 16
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K / 2)
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES) //=4
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP) //=8

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 8
#define WARP_COL_TILES 4

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)


#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

#define ITEMS_PER_INT4 32 //128/4
#define ITEMS_PER_INT 8 //32/4

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 2 "int4" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_INT4 2

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

__global__ void mm_wu4au4(const int4 *A, const int4 *B, int *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {

  int M_TILES = M_GLOBAL / M;
  int N_TILES = N_GLOBAL / N;
  int K_TILES = K_GLOBAL / K;

  extern __shared__ int4 shmem[][CHUNK_K * K / ITEMS_PER_INT4 + SKEW_INT4];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C or D matrix tiles this warp computes.
  int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
                               (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * M * WARP_COL_TILES +
                               (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  int *shmem_warp_stream_ptr =
      (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * (BLOCK_COL_TILES / WARPS_PER_BLOCK) * M;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }


    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    // Init C
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
          wmma::fill_fragment(c[i][j], 0);
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const int4 *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL / ITEMS_PER_INT4] +
                                           M * K_GLOBAL / ITEMS_PER_INT4 * (warpId % 4) * (BLOCK_COL_TILES / 4))
                                        : (&B[block_tile_j * N * K_GLOBAL / ITEMS_PER_INT4] +
                                           N * K_GLOBAL / ITEMS_PER_INT4 * (warpId % 4) * (BLOCK_ROW_TILES / 4));

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * (BLOCK_COL_TILES / (WARPS_PER_BLOCK / 2)))
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * (BLOCK_ROW_TILES / (WARPS_PER_BLOCK / 2)) + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = ((int4 *)warp_ptr + tile_k * K / ITEMS_PER_INT4 +
                       (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL / ITEMS_PER_INT4) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;
      
      // CHUNK_COPY_LINES_PER_WARP=4, loop iter=8
#pragma unroll
      for (int i = 0; i < (BLOCK_COL_TILES / (WARPS_PER_BLOCK / 2)) * M / CHUNK_COPY_LINES_PER_WARP; i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr += K_GLOBAL * CHUNK_COPY_LINES_PER_WARP / ITEMS_PER_INT4;
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::u4, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::u4, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / BLOCK_ROW_WARPS) * M * WARP_COL_TILES + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step * K / ITEMS_PER_INT4];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_INT4 * ITEMS_PER_INT4);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % BLOCK_ROW_WARPS) + (j * N);
              const int4 *tile_ptr = &shmem[shmem_idx_b][k_step * K / ITEMS_PER_INT4];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_INT4 * ITEMS_PER_INT4);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {

        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * M + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
       (block_tile_i + warpId * (BLOCK_COL_TILES / WARPS_PER_BLOCK)) * M * N_GLOBAL + block_tile_j * N;
    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < (BLOCK_COL_TILES / WARPS_PER_BLOCK) * M; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + N_GLOBAL * i) + laneId) =
          *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

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
    int mask = 15;  //0b00001111
  
    for (int m = 0; m < M_GLOBAL; m++) {
        for (int n = 0; n < N_GLOBAL; n++) {
            int tmp = 0;
            for(int k_tile = 0; k_tile < K_GLOBAL/8; k_tile++) {
                int w_int = W_int[m*K_GLOBAL/8 + k_tile];
                int x_int = X_int[n*K_GLOBAL/8 + k_tile];
                for(int k = 0; k < 8; k++) {
                    int shift = k*4;
                    int x_val = ((mask << shift) & x_int) >> shift;
                    int w_val = ((mask << shift) & w_int) >> shift;
		    if(x_val<0 || w_val<0)
		        printf("cpu compute error");
                    tmp += x_val * w_val;
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
  //checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  int d;
  cudaError_t err = cudaGetDevice(&d);
  if (err != cudaSuccess) 
      printf("kernel cuda error: %d, %s\n", (int)err, cudaGetErrorString(err));
  printf("device = %d\n", d);
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, d));

  int X_BIT = 4;
  int W_BIT = 4;

  //int M_GLOBAL = 64;
  //int M_GLOBAL = 1024;
  // int N_GLOBAL = 64;
  // int K_GLOBAL = 128;
  //for (int N_GLOBAL=128; N_GLOBAL<=2048; N_GLOBAL += 128 ) {
  for (int N_GLOBAL=512; N_GLOBAL<=32768; N_GLOBAL *= 2 ) {
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
      mm_wu4au4, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
  
    printf("number of SMs: %d\n", deviceProp.multiProcessorCount);
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
              (mm_wu4au4<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                    SHMEM_SZ>>>(W, X, Output, M_GLOBAL, N_GLOBAL, K_GLOBAL)));
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
