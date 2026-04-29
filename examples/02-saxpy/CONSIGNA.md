# Ejercicio 2 — SAXPY

**SAXPY** (`Single-precision A*X Plus Y`) es uno de los kernels más usados en
álgebra lineal: `Y = a*X + Y`. Forma parte del estándar BLAS y es el "hola
mundo" después de la suma de vectores.

## Objetivos pedagógicos

- Pasar un escalar (`a`) como argumento al kernel
- Operar sobre un vector que se lee **y** se escribe (`Y`)
- Comparar métricas con el ejercicio 01

## Pasos

1. Abrir `saxpy.cu` y completar los tres `TODO`.
2. Compilar y simular: `make run`.
3. Verificar: `Y[i]` debe valer `4.0` para todo `i` (con `a=2`, `X=1`, `Y₀=2`).

## Para el informe

Después de correr el kernel, GPGPU-Sim genera un reporte. Buscar y reportar:

| Métrica | Dónde aparece en el log | Qué significa |
|---------|--------------------------|----------------|
| `gpu_sim_cycle` | Final del log | Ciclos totales de simulación |
| `gpu_ipc` | Final del log | Instrucciones por ciclo (eficiencia) |
| `gpgpu_n_load_insn` | Stats globales | Cantidad de loads ejecutados |
| `L1D_total_cache_miss_rate` | Cache stats | Tasa de fallos en cache L1 de datos |

**Pregunta**: ¿Cuántos loads hace SAXPY por elemento vs. cuántos hace
`vector_add`? ¿Por qué? Justificar mirando el código.

## Pista (si están perdidos)

El esqueleto es casi idéntico al de `01-vector-add`. La única diferencia es
que **leen Y antes de escribirlo** y multiplican `X` por el escalar.
