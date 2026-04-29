// Ejercicio 4: Reduccion paralela (suma de un vector)
//
// Calcula: total = sum(input[0..N])
//
// Es el ejercicio mas dificil de los 4. Aqui aparecen los conceptos
// fundamentales de programacion paralela en GPU:
//
//   - SHARED MEMORY: memoria compartida entre threads del mismo bloque,
//     mucho mas rapida que la global.
//   - SINCRONIZACION: __syncthreads() para que todos los threads del bloque
//     terminen una fase antes de empezar la siguiente.
//   - DIVERGENCIA DE WARPS: cuando threads del mismo warp toman caminos
//     distintos en un if, el hardware ejecuta ambos serialmente.
//   - REDUCCION EN ARBOL: tecnica clasica para sumar N elementos en
//     log2(N) pasos en lugar de N pasos.
//
// CONSIGNA:
//   1. Completar el kernel de reduccion en arbol con shared memory
//   2. El host suma los resultados parciales de cada bloque (esto esta
//      hecho por simplicidad; en kernels reales se hace en GPU tambien)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024              // Tamanio del vector
#define BLOCK_SIZE 256       // Threads por bloque

// TODO: Implementar reduccion en arbol con shared memory
//
// Esquema:
//   1. Cada thread carga un elemento de input a shared memory (sdata[tid])
//   2. __syncthreads()
//   3. Loop: stride = blockDim.x/2, blockDim.x/4, ..., 1
//        Si tid < stride:
//          sdata[tid] += sdata[tid + stride]
//        __syncthreads()
//   4. El thread 0 escribe sdata[0] a output[blockIdx.x]
__global__ void reduce(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    // TU CODIGO AQUI
    // Pista: int tid = threadIdx.x;
    //        int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

int main(void) {
    const size_t bytes = N * sizeof(float);
    float *h_input = (float*)malloc(bytes);

    // Vector de unos -> la suma debe dar exactamente N
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *h_partial = (float*)malloc(blocks * sizeof(float));

    float *d_input, *d_partial;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // El tercer parametro <<<...,...,SIZE>>> es el tamanio de shared memory
    reduce<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_partial, N);

    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Suma final en CPU (por simplicidad)
    float total = 0.0f;
    for (int i = 0; i < blocks; i++) total += h_partial[i];

    if (total == (float)N) {
        printf("OK: suma = %.0f (esperado %d)\n", total, N);
    } else {
        printf("FALLA: suma = %.0f (esperado %d)\n", total, N);
    }

    cudaFree(d_input); cudaFree(d_partial);
    free(h_input); free(h_partial);
    return 0;
}
