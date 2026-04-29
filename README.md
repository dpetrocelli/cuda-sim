# CUDA sin GPU — Simulador para SD 2026

Entorno Docker con **GPGPU-Sim 4.2.1** (febrero 2025) para escribir, compilar y
ejecutar código CUDA **sin necesidad de tener una placa NVIDIA**. Pensado para
la cátedra de **Sistemas Distribuidos y Programación Paralela (SD 2026)** —
UNLu DCB — mientras esperamos el cluster Kubernetes con GPUs reales.

> **¿Sos estudiante?** Empezá por [`docs/GUIA.md`](docs/GUIA.md). Ahí está
> todo: qué es esto, por qué se usa, cómo se setea, cómo se interpreta la
> salida del simulador y cómo se entregan los TPs.

## ¿Qué es GPGPU-Sim?

Simulador *cycle-accurate* de GPUs NVIDIA desarrollado en UBC. Ejecuta binarios
CUDA reales, intercepta las llamadas a la API y simula la ejecución en una GPU
virtual (Volta, Turing, Pascal, etc.). Devuelve métricas detalladas:

- Ciclos totales y ocupación de cada SM
- Hits/misses de cache L1/L2
- Transacciones de memoria global
- Divergencia de warps
- Uso de unidades funcionales

Es el mismo simulador que usan los papers de arquitectura GPU. Para clase es
ideal porque los pibes **ven por qué** un kernel es lento — algo que con GPU
real queda oculto detrás del driver.

## Requisitos

- Docker + Docker Compose
- ~2 GB de disco para la imagen
- Cualquier OS (Linux, Mac, Windows con WSL2)

## Quick start

```bash
# 1. Buildear la imagen (primera vez, tarda ~10-15 min)
docker compose build

# 2. Entrar al container
docker compose run --rm cuda-sim

# 3. Adentro del container, ir a un ejemplo y correrlo
cd 01-vector-add
make run
```

## Estructura

```
cuda-sim/
├── Dockerfile              # Imagen con GPGPU-Sim 4.2.1 sobre CUDA 11.7
├── docker-compose.yml      # Servicio listo para usar
├── docs/
│   └── GUIA.md             # Guía completa para estudiantes
├── examples/
│   ├── 01-vector-add/      # RESUELTO. Hello world de CUDA.
│   ├── 02-saxpy/           # A COMPLETAR. Y = a*X + Y.
│   ├── 03-matrix-mul/      # A COMPLETAR. Multiplicación de matrices.
│   └── 04-reduction/       # A COMPLETAR. Reducción con shared memory.
└── README.md
```

Cada ejercicio tiene su `CONSIGNA.md` con los objetivos pedagógicos, los
pasos a seguir y las métricas a reportar.

## ¿Qué GPU se simula?

Por defecto **Tesla V100 (Volta, SM7_QV100)**. Para cambiar, editar la variable
`GPU_CONFIG` del Makefile:

| Config | GPU emulada | Año | Uso |
|--------|-------------|-----|-----|
| `SM7_QV100` | Tesla V100 | 2017 | Default — buena para clase |
| `SM75_RTX2060` | RTX 2060 | 2019 | Turing, primera con RT cores |
| `SM6_TITANX` | Titan X Pascal | 2016 | Pascal, más simple |

## Importante: rendimiento

El simulador es **lentísimo** (es ciclo-a-ciclo). Un kernel que en GPU real
tarda 1ms puede tardar varios minutos. Por eso los ejemplos usan tamaños chicos
(N=1024 elementos). Esto es **normal** y deseable para enseñar.

## Migración a GPU real (cuando llegue el cluster)

Cuando tengamos el cluster K8s con GPUs físicas:

1. Descomentar la sección `deploy.resources` en `docker-compose.yml`.
2. Cambiar la base del Dockerfile a `nvidia/cuda:12.x-devel-ubuntu22.04` (sin
   instalar GPGPU-Sim).
3. En el Makefile, sacar `--cudart shared` para que use el driver real.

El código `.cu` de los pibes **no cambia**. Mismo binario corre en simulador o
GPU real.

## Troubleshooting

**`docker compose build` falla en el `make` de GPGPU-Sim.**
Suele ser por versión de GCC. La imagen base usa GCC 9 que es la testeada.
Verificar que no se cambió la base.

**El kernel "no hace nada".**
Asegurarse de copiar la config con `make config` antes de ejecutar (o usar
`make run` que lo hace automáticamente). Sin `gpgpusim.config` en el cwd, la
ejecución falla silenciosa o usa el driver real (que no tenemos).

**Lentitud extrema.**
Reducir `N` en el ejemplo. GPGPU-Sim no es para datasets grandes.

## Referencias

- [GPGPU-Sim oficial](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
- [Manual de GPGPU-Sim](http://www.gpgpu-sim.org/manual/index.php/Main_Page)
- [Accel-Sim](https://accel-sim.github.io/) (alternativa moderna trace-driven)
