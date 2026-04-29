# Ejercicio 3 — Multiplicación de matrices

`C = A × B` con matrices `N×N`. Es el "Hello World" de GPU computing serio:
aparece en redes neuronales, simulaciones físicas, gráficos 3D.

## Objetivos pedagógicos

- Configurar **grid 2D**: bloques y threads en `(x, y)`
- Entender el **patrón de acceso a memoria** y por qué importa
- Ver el costo de **no reusar datos** (cada elemento de A se lee N veces)

## Pasos

1. Completar el kernel `matmul()`. Tip: cada thread calcula **un elemento** de
   `C`, recorriendo una fila de `A` y una columna de `B`.
2. Configurar el grid 2D. Si usan bloques de 16×16 y `N=32`, son 2×2 = 4
   bloques.
3. Verificar: con `A = I` (identidad) y `B = unos`, debe dar `C = unos`.

## Para el informe

| Métrica | Qué mirar |
|---------|-----------|
| `gpu_sim_cycle` | ¿Cuánto más tarda que `vector_add`? |
| `gpgpu_n_load_insn` | ¿Cuántos loads totales? Comparar con `2 * N³` esperado |
| `L1D_total_cache_miss_rate` | Acá viene lo interesante |

**Pregunta clave**: ¿Por qué la tasa de miss de L1 es alta en este kernel
naive? Pista: pensar el orden en que cada thread del bloque accede a `B`.

## Bonus (recomendado)

Implementar una **segunda versión** del kernel con **shared memory** (tile).
Comparar las métricas. Deberían ver:

- Caída de loads a memoria global
- Caída de miss rate de L1
- Reducción de ciclos totales

Esto es **el** ejemplo canónico para mostrar el valor de la jerarquía de
memoria en GPU. Está en cualquier libro de CUDA (Kirk & Hwu cap. 4).
