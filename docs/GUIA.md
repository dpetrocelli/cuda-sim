# Guía del estudiante — CUDA con GPGPU-Sim

> **Para qué es esta guía**: aprender a programar para GPUs **sin tener una
> placa NVIDIA**. Usamos un simulador que se llama GPGPU-Sim, empaquetado en
> Docker para que funcione en cualquier laptop.

---

## 1. ¿Qué es GPGPU-Sim y por qué lo usamos?

### Qué es

**GPGPU-Sim** es un simulador *cycle-accurate* de GPUs NVIDIA, desarrollado
en la University of British Columbia. Es el simulador que usan los papers
académicos de arquitectura de GPUs hace más de 15 años.

> *Cycle-accurate* significa que reproduce la ejecución de cada instrucción
> ciclo por ciclo. No es un emulador rápido, es un modelo del hardware.

Ejecuta **binarios CUDA reales**: ustedes escriben un `.cu` normal, lo
compilan con `nvcc`, y al correrlo el simulador intercepta las llamadas a la
API CUDA y las ejecuta sobre una GPU virtual (Tesla V100, RTX 2060, etc.).

### Por qué lo usamos en clase

1. **No necesitamos hardware**. Mientras esperamos el cluster Kubernetes con
   GPUs reales, todos pueden trabajar desde su laptop.
2. **Visibilidad total**. El simulador devuelve métricas que en una GPU real
   están escondidas atrás del driver: ciclos por instrucción, hits/misses de
   cache, divergencia de warps, ocupancia de los SMs.
3. **Reproducibilidad**. Todos los pibes corren la misma "GPU" — mismas
   métricas, mismos resultados. No hay variación por hardware.
4. **Pedagógicamente superior**. En GPU real un kernel "anda y ya". En el
   simulador ven *por qué* anda mejor o peor.

### Limitaciones (importante saberlas)

- **Lentísimo**. Un kernel que en GPU real tarda 1ms acá puede tardar varios
  minutos. Por eso los ejercicios usan tamaños chicos (N=1024).
- **No sirve para entrenar redes** (no corran PyTorch encima — no es para eso).
- **CUDA limitada**. La versión usada es CUDA 11.7. Las features muy nuevas
  (ej. CUDA Graphs avanzados) pueden no estar.

---

## 2. Setup

### Requisitos

- Una laptop con Docker + Docker Compose instalado.
- Linux, Mac o Windows (con WSL2). No importa cuál.
- ~3 GB de disco libre.
- Conexión para la primera vez (descarga la imagen base de NVIDIA).

### Primera vez (build de la imagen)

```bash
git clone <URL del repo>
cd cuda-sim
docker compose build
```

El build tarda **10–15 minutos** la primera vez. Es porque compila GPGPU-Sim
desde fuente. Una vez hecho, queda en caché y no se vuelve a hacer.

### Día a día (entrar al entorno)

```bash
docker compose run --rm cuda-sim
```

Esto los deja adentro del container, en `/workspace`, con acceso a los
ejemplos. Cualquier cambio que hagan a los `.cu` desde su editor en la
máquina host se ve adentro del container automáticamente (volume mount).

---

## 3. Anatomía de un programa CUDA

```c
__global__ void miKernel(float *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= 2.0f;
}

int main() {
    float *h_data = malloc(...);          // 1. memoria en CPU (host)
    float *d_data; cudaMalloc(&d_data,...);  // 2. memoria en GPU (device)
    cudaMemcpy(d_data, h_data, ..., HostToDevice);  // 3. copiar a GPU
    miKernel<<<bloques, threads>>>(d_data, N);       // 4. lanzar kernel
    cudaMemcpy(h_data, d_data, ..., DeviceToHost);   // 5. traer resultado
    cudaFree(d_data); free(h_data);                  // 6. liberar
}
```

### Conceptos mínimos

| Concepto | Qué es |
|---------|--------|
| **Thread** | Una unidad de ejecución. En GPU son miles. |
| **Block** | Grupo de threads que comparten *shared memory* y se sincronizan entre sí. |
| **Grid** | Conjunto de blocks que ejecutan el mismo kernel. |
| **Warp** | 32 threads que ejecutan en lock-step (SIMT). El warp es la unidad real de scheduling. |
| **SM** | *Streaming Multiprocessor*. La unidad de ejecución física de la GPU. Una V100 tiene 80. |
| **Shared memory** | Memoria muy rápida, compartida dentro de un block (~96 KB por SM en Volta). |
| **Global memory** | Memoria principal de la GPU (HBM2). Grande pero lenta. |

### Modelo de programación SIMT

CUDA es **SIMT** (*Single Instruction Multiple Thread*), una variante de SIMD.
Todos los threads de un warp ejecutan **la misma instrucción** sobre datos
distintos. Si un `if` hace que algunos threads tomen una rama y otros la
contraria, el warp **diverge**: el hardware ejecuta ambas ramas serialmente,
desactivando los threads que no aplican. Esto se llama *warp divergence* y es
una de las principales causas de pérdida de rendimiento.

---

## 4. Ejercicios

| # | Tema | Concepto principal |
|---|------|---------------------|
| 01 | Suma de vectores | Hello world. Indexado 1D. |
| 02 | SAXPY | Pasaje de escalares. Lectura+escritura. |
| 03 | Multiplicación de matrices | Grid 2D. Coalescing. Reuso de datos. |
| 04 | Reducción | Shared memory. Sincronización. Divergencia. |

El **ejercicio 01 está resuelto** como referencia. Los **02, 03 y 04 los
implementan ustedes** siguiendo la consigna en cada carpeta (`CONSIGNA.md`).

### Cómo correr cada ejercicio

```bash
# Adentro del container
cd 01-vector-add   # o 02-saxpy, 03-..., 04-...
make run
```

`make run` hace tres cosas:
1. Compila con `nvcc` linkeando contra la `libcudart` de GPGPU-Sim.
2. Copia la config de la GPU simulada (`gpgpusim.config`) al directorio.
3. Ejecuta el binario, que dispara la simulación.

---

## 5. Cómo leer la salida del simulador

Cuando corren `make run`, GPGPU-Sim imprime **mucho output**. Lo importante
está al final del log, en una sección que empieza con
`GPGPU-Sim: ** STATS **`.

### Métricas claves

```
gpu_sim_cycle = 12345              # Ciclos totales de la GPU
gpu_tot_sim_insn = 67890           # Instrucciones totales ejecutadas
gpu_ipc = 5.50                     # Instrucciones Por Ciclo (mas alto = mejor)
gpu_occupancy = 87.5%              # Ocupancia de los SMs
```

#### Cache stats

```
L1D_total_cache_accesses = 1024
L1D_total_cache_misses = 32
L1D_total_cache_miss_rate = 0.0312
```

L1D = cache de datos L1. Una **miss rate baja** indica que el patrón de
acceso es bueno (coalesced).

#### Memoria global

```
gpgpu_n_load_insn = 2048           # Cantidad de loads
gpgpu_n_store_insn = 512           # Cantidad de stores
gpgpu_n_mem_read_global = ...
```

### Lo que hay que reportar en cada TP

Para cada ejercicio:

1. **Tabla con las métricas** anteriores.
2. **Comparación con el ejercicio anterior** (¿más ciclos? ¿peor IPC?).
3. **Una explicación** de por qué se ven esos números mirando el código.
4. **Capturas de pantalla** del log para evidencia.

---

## 6. ¿Qué pasa con la GPU real?

Cuando llegue el cluster Kubernetes con GPUs físicas:

1. El **mismo código `.cu` corre tal cual**. No cambia ni una línea.
2. En el `Dockerfile`, descomentamos la sección de `nvidia/cuda` runtime
   y deshabilitamos GPGPU-Sim.
3. Comparamos: las métricas reales contra las simuladas. Ahí ven la
   precisión del simulador (suele estar dentro del 10-20% del comportamiento
   real para arquitecturas soportadas).

---

## 7. Recursos

### Lectura obligatoria

- **Kirk & Hwu**, *Programming Massively Parallel Processors* (3ra ed.). El
  libro de cabecera.
- Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA whitepaper).
  Mostra 7 versiones del mismo kernel cada vez más optimizadas.

### Documentación

- [GPGPU-Sim manual](http://www.gpgpu-sim.org/manual/index.php/Main_Page)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Repo GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution)

### Comunidad

- Stack Overflow tag `cuda`
- NVIDIA Developer Forum

---

## 8. Troubleshooting

| Problema | Solución |
|---------|----------|
| `docker compose build` falla en el make | Verificar que la base es Ubuntu 20.04 y CUDA 11.7. Otras combinaciones fallan. |
| El binario "no hace nada" | Olvidaron el `gpgpusim.config` en el cwd. `make run` lo copia automáticamente. |
| Lentitud extrema | Reducir `N`. Es esperable que sea lento. |
| `nvcc: command not found` | No están adentro del container. Correr `docker compose run --rm cuda-sim` primero. |
| Cambios al `.cu` no se ven | Verificar que el volume mount funciona. `ls /workspace` dentro del container debe listar los archivos. |

---

## 9. Entrega

Cada ejercicio entregado debe incluir:

- El `.cu` completado y funcionando.
- Un `RESULTADOS.md` con la tabla de métricas, capturas y análisis.
- Estar en el repo del grupo, en una rama por ejercicio.

**Buena suerte 🚀**
