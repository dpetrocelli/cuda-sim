// Ejercicio 3: Multiplicacion de matrices  C = A x B
//
// Aqui se ven por primera vez los conceptos de:
//   - Indexado 2D con threadIdx.x/y, blockIdx.x/y, blockDim.x/y
//   - Coalescing de memoria: el orden en que threads acceden a memoria
//     impacta DRAMATICAMENTE el rendimiento
//   - Reuso de datos: cada elemento de A y B se lee N veces (oportunidad
//     para usar shared memory en una version optimizada)
//
// CONSIGNA:
//   1. Implementar el kernel naive (sin shared memory)
//   2. Lanzarlo con grid 2D
//   3. (BONUS) Implementar version con shared memory y comparar metricas

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 32  // Matriz NxN. Pequenio porque GPGPU-Sim es lento.

// TODO 1: Kernel naive
// C[row][col] = sum_{k=0..N} A[row][k] * B[k][col]
__global__ void matmul(const float *A, const float *B, float *C, int n) {
    // TU CODIGO AQUI
    // Pista: dos indices (row, col), un loop interno sobre k
}

int main(void) {
    const size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // A = identidad, B = matriz de unos -> C debe ser matriz de unos
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            h_A[i*N + j] = (i == j) ? 1.0f : 0.0f;
            h_B[i*N + j] = 1.0f;
        }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // TODO 2: Configurar grid 2D y lanzar el kernel
    // dim3 threads(BLOCK, BLOCK);
    // dim3 grid(...);
    // matmul<<<grid, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // TODO 3: Verificar que C sea matriz de unos
    // Para A=I, B=ones: C debe ser ones.

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
