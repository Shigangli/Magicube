#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

// GPU configuration.
#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 8
#define N 8
#define K 32

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 1

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 2

#define WARP_ROW_TILES 1
#define WARP_COL_TILES 1

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

__global__ void apmm_wu4au4(const int4 *W, const int4 *X, int *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int wb, int xb) {
  // GEMM configuration.
  int K_TILES = K_GLOBAL / 32;

  int ROW_BIT = K_GLOBAL / 32;

  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {

    //compute a 16x16 block of D
    const unsigned int block_tile_i = block_pos / (N_GLOBAL/16) * 16;
    const unsigned int block_tile_j = block_pos % (N_GLOBAL/16) * 16;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_GLOBAL) {
      break;
    }

    wmma::fragment<wmma::accumulator, M, N, K, int> c;

    wmma::fill_fragment(c, 0);

    const int4 *warp_ptr_w;
    const int4 *warp_ptr_x;
    warp_ptr_w = &W[block_tile_i * ROW_BIT + 8*(warpId/2) * ROW_BIT + warpId%2];
    warp_ptr_x = &X[block_tile_j * ROW_BIT + 8*(warpId/2) * ROW_BIT + warpId%2];

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += 2) {
      // Offset in shared memory from which the B matrix is stored.
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.

      int *shmem_ptr = (int*)shmem + warpId*8*4*(CHUNK_K+SKEW) + (laneId/4)*4*(CHUNK_K+SKEW) + laneId%4;

      int *lane_ptr_w = (int*)warp_ptr_w + laneId/4*ROW_BIT*4 + laneId%4 + tile_k*4;
      int *lane_ptr_x = (int*)warp_ptr_x + laneId/4*ROW_BIT*4 + laneId%4 + tile_k*4;
      
      *shmem_ptr = *lane_ptr_w;
      shmem_ptr += 8*4*WARPS_PER_BLOCK;
      *shmem_ptr = *lane_ptr_x;

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
      for (int k_step = 0; k_step < 2; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::u4, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, M, N, K, precision::u4, wmma::col_major> b;

        size_t shmem_idx_w = (warpId / 2) * 2 * M + k_step * M;
        const int4 *tile_ptr_w = &shmem[shmem_idx_w][0];
        wmma::load_matrix_sync(a, tile_ptr_w, (CHUNK_K + SKEW)*32);

        size_t shmem_idx_x = 32 + (warpId % 2) * 2 * N + k_step * N;
        const int4 *tile_ptr_x = &shmem[shmem_idx_x][0];
        wmma::load_matrix_sync(b, tile_ptr_x, (CHUNK_K + SKEW)*32);
        wmma::mma_sync(c, a, b, c);
      }
      __syncthreads();
    }

    size_t gmem_idx = block_tile_i*N_GLOBAL + block_tile_j + N_GLOBAL * (warpId / BLOCK_ROW_WARPS) * M + (warpId % BLOCK_ROW_WARPS) * N;

    // Now that shared memory contains all the D tiles, stream them to global memory.
    int *tile_ptr = &D[gmem_idx];
    wmma::store_matrix_sync(tile_ptr, c, N_GLOBAL, C_LAYOUT);
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
      apmm_wu4au4, cudaFuncAttributeMaxDynamicSharedMemorySize,
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
              (apmm_wu4au4<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
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
