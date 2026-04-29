# Ejercicio 5 — Accel-Sim sobre trazas SASS

## Qué cambia respecto a los ejercicios 01-04

Los ejercicios 01-04 usan **GPGPU-Sim** en modo *execution-driven*: ustedes
escriben un `.cu`, lo compilan y al ejecutar el binario el simulador
intercepta las llamadas CUDA y las simula ciclo por ciclo. Es lento pero
permite escribir su propio código.

Este ejercicio usa **Accel-Sim** en modo *trace-driven*. La filosofía es
distinta:

1. **NVIDIA NVBit** captura una traza SASS (instrucciones reales del ISA
   nativo de la GPU) ejecutando el binario sobre **una GPU física**.
2. La traza se distribuye como un archivo.
3. El simulador **Accel-Sim** consume esa traza y simula la
   microarquitectura sobre ella.

### Ventajas

- Mucho **más rápido** que execution-driven (ya tiene las instrucciones
  SASS, no tiene que ejecutar el código).
- Soporta **arquitecturas más nuevas** (Turing, Ampere) que GPGPU-Sim no
  modela bien todavía.
- Permite simular **binarios sin código fuente** (mientras tengan la traza).

### Limitaciones

- Necesitan una traza pre-grabada (no pueden escribir su propio kernel
  arbitrario sin GPU para capturar la traza).
- El estado del programa es "input-fixed": la traza es lo que pasó esa
  vez. No pueden cambiar los inputs sin re-trazar.

## Ejecución

```bash
# Desde el repo cuda-sim, en el host:
docker compose run --rm accel-sim bash

# Adentro del container:
cd /workspace
bash run.sh
```

El script clona Accel-Sim, lo buildea (~10 minutos la primera vez),
descarga una traza de **vectoradd** de Rodinia 2.0 capturada sobre Tesla
V100, y simula la microarquitectura.

## Para el informe

1. **Salida final** del simulador. Los campos clave son los mismos que en
   GPGPU-Sim (`gpu_sim_cycle`, `gpu_ipc`, cache stats).

2. **Comparación con el ejercicio 01**: ambos simulan `vectoradd` sobre
   V100. ¿Dan los mismos números? ¿Por qué la diferencia?

   - Pista: el ejercicio 01 usa **PTX** (virtual ISA), Accel-Sim usa
     **SASS** (machine ISA real). PTX se traduce a SASS por el JIT del
     driver — pueden diferir.

3. **Tiempo de wall-clock** de la simulación. Comparen con el ejercicio
   01. La diferencia debería ser notable.

4. **Una reflexión**: para qué tipo de proyecto elegirían cada uno.

## Recursos

- [Accel-Sim framework](https://accel-sim.github.io/)
- Paper: Khairy et al., *Accel-Sim: An Extensible Simulation Framework for
  Validated GPU Modeling*, ISCA 2020.
- [Trazas pre-grabadas](https://engineering.purdue.edu/tgrogers/accel-sim/traces/)
  (Purdue mirror).
