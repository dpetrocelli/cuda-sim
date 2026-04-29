// Ejercicio 2: SAXPY  ->  Y = a*X + Y
//
// SAXPY = "Single-precision A*X Plus Y". Es uno de los kernels mas usados en
// algebra lineal (BLAS Level 1). Aqui lo implementan ustedes.
//
// CONCEPTOS A APRENDER:
//   - Kernel con multiples operaciones por thread
//   - Pasaje de un escalar (a) al kernel
//   - Diferencia entre lectura/escritura: Y se lee Y se escribe
//
// CONSIGNA:
//   1. Completar el kernel saxpy()
//   2. Lanzarlo con la configuracion adecuada
//   3. Verificar el resultado en CPU
//   4. Reportar en el informe: ciclos totales, ocupancia, hits L1

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// TODO 1: Implementar el kernel SAXPY
// Cada thread debe calcular: y[i] = a * x[i] + y[i]
__global__ void saxpy(int N, float a, const float *x, float *y) {
    // TU CODIGO AQUI
    // Pista: usar blockIdx, blockDim, threadIdx para calcular el indice
}

int main(void) {
    const int N = 1024;
    const float a = 2.0f;
    const size_t bytes = N * sizeof(float);

    // Host
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Device
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // TODO 2: Configurar grid y bloques, lanzar el kernel
    // saxpy<<<..., ...>>>(N, a, d_x, d_y);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    // TODO 3: Verificar. Y[i] esperado = a * 1.0 + 2.0 = 4.0
    // Imprimir OK o cantidad de errores.

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return 0;
}
