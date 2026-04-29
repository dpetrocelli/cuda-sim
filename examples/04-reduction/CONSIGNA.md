# Ejercicio 4 — Reducción paralela

Sumar un vector en GPU. Suena trivial en CPU (`for + acumulador`), pero en GPU
es el ejercicio canónico para entender la **jerarquía de memoria** y la
**sincronización**. Es el más difícil de los 4.

## Objetivos pedagógicos

- **Shared memory**: declarar y usar `__shared__`
- **Sincronización**: `__syncthreads()` y por qué es crítico
- **Reducción en árbol**: pasar de O(N) a O(log N) en paralelo
- **Divergencia de warps**: el costo de los `if` que dividen al warp

## El algoritmo (reducción en árbol)

```
Inicio:    [1][1][1][1][1][1][1][1]   (8 elementos)
Paso 1 (stride=4):
  threads 0-3 hacen sdata[i] += sdata[i+4]
           [2][2][2][2][1][1][1][1]
Paso 2 (stride=2):
  threads 0-1 hacen sdata[i] += sdata[i+2]
           [4][4][2][2][1][1][1][1]
Paso 3 (stride=1):
  thread 0 hace sdata[0] += sdata[1]
           [8][4][2][2][1][1][1][1]

Resultado: sdata[0] = 8
```

En `log2(N)` pasos en lugar de `N`. Para `N=256` son 8 pasos paralelos vs 256
secuenciales.

## Pasos

1. Completar el kernel siguiendo el esquema documentado en `reduction.cu`.
2. Compilar y simular: `make run`.
3. Verificar: con vector de unos y `N=1024`, la suma debe dar exactamente
   `1024`.

## Para el informe

| Métrica | Qué mostrar |
|---------|-------------|
| `gpu_sim_cycle` | Total de ciclos |
| `gpgpu_n_shmem_insn` | Operaciones sobre shared memory |
| `gpgpu_n_load_insn` | Loads a memoria global |
| `warp_divergence` (si aparece) | Veces que un warp se divergió |

**Preguntas clave**:

1. En el loop de reducción, el `if (tid < stride)` hace que algunos threads
   trabajen y otros no. ¿En qué momento empieza a haber divergencia de
   warps? Recordar que un warp = 32 threads.
2. ¿Qué pasaría si quitamos el `__syncthreads()` después de cada paso?
3. ¿Cuántos accesos a memoria **global** hace este kernel vs. uno que sume
   sin shared memory?

## Bibliografía sugerida

- Mark Harris, *"Optimizing Parallel Reduction in CUDA"* (NVIDIA whitepaper).
  Es **el** documento clásico — muestra 7 versiones progresivamente
  optimizadas del mismo kernel. Lectura obligatoria para entender CUDA en
  serio.
