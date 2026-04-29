// Ejemplo 1: Suma de vectores en GPU
// Calcula C[i] = A[i] + B[i] en paralelo.
//
// Conceptos que se ven aqui:
//   - Kernel CUDA (__global__)
//   - Indexacion con threadIdx + blockIdx + blockDim
//   - Transferencias host <-> device (cudaMemcpy)
//   - Configuracion de grid y bloques
//
// Compilacion: make
// Ejecucion en simulador: make run

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel: cada thread calcula un elemento del vector resultado
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // Tamanio chico a proposito: GPGPU-Sim simula ciclo a ciclo y es lento.
    // Con 1024 elementos el ejemplo termina en segundos.
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // 1. Reservar memoria en host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // 2. Inicializar vectores de entrada
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 3. Reservar memoria en device (GPU simulada)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 4. Copiar datos host -> device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 5. Configurar grid: 256 threads por bloque, los bloques que hagan falta
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Lanzando kernel: %d bloques x %d threads = %d threads totales para N=%d\n",
           blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock, N);

    // 6. Lanzar el kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 7. Copiar resultado device -> host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 8. Verificar resultado
    int errores = 0;
    for (int i = 0; i < N; i++) {
        float esperado = h_A[i] + h_B[i];
        if (h_C[i] != esperado) errores++;
    }
    if (errores == 0) {
        printf("OK: %d elementos correctos. C[0]=%.1f, C[N-1]=%.1f\n",
               N, h_C[0], h_C[N-1]);
    } else {
        printf("FALLA: %d errores\n", errores);
    }

    // 9. Liberar
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return errores == 0 ? 0 : 1;
}
