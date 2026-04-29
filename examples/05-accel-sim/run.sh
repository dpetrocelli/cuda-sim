#!/usr/bin/env bash
# Ejercicio 5 — Accel-Sim sobre trazas SASS pre-grabadas
#
# Este script se ejecuta DENTRO del container accel-sim
# (ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8).
#
# Flujo:
#   1. Clonar el repo de Accel-Sim si no esta presente.
#   2. Buildear el simulador.
#   3. Descargar trazas SASS de ejemplo.
#   4. Correr una simulacion de "vectoradd" sobre Tesla V100 (SM7_QV100).
#
# La diferencia con el flujo de los ejercicios 01-04 (GPGPU-Sim) es que aqui
# NO compilan ningun .cu ustedes — usan trazas que NVIDIA-NVBit ya capturo
# de una GPU real ejecutando el binario. Accel-Sim consume esas trazas y
# simula la microarquitectura sobre ellas.

set -euo pipefail

ACCEL_DIR=/opt/accel-sim
TRACES_DIR=/opt/accel-sim-traces

# 1) Clonar y buildear (si no esta listo)
if [[ ! -d "$ACCEL_DIR/gpu-simulator" ]]; then
  echo "==> Clonando Accel-Sim framework..."
  git clone https://github.com/accel-sim/accel-sim-framework.git "$ACCEL_DIR"
fi

cd "$ACCEL_DIR"
if [[ ! -x ./gpu-simulator/bin/release/accel-sim.out ]]; then
  echo "==> Buildeando simulador (puede tardar varios minutos)..."
  pip3 install -r requirements.txt
  source ./gpu-simulator/setup_environment.sh
  make -j -C ./gpu-simulator/
fi

# 2) Descargar trazas de ejemplo (rodinia-3.1 / vectoradd) si faltan
if [[ ! -d "$TRACES_DIR" ]]; then
  echo "==> Descargando trazas SASS de ejemplo..."
  mkdir -p "$TRACES_DIR"
  cd "$TRACES_DIR"
  # El proyecto distribuye trazas en un script interactivo.
  # Aqui descargamos una traza pequenia directa para no depender del prompt.
  wget -q --show-progress \
    https://engineering.purdue.edu/tgrogers/accel-sim/traces/tested-cfgs/SM7_QV100/11.0/rodinia_2.0-ft/vectoradd-100_100/traces.tgz \
    -O traces.tgz || {
      echo "Fallo la descarga directa. Caer al script interactivo:"
      cd "$ACCEL_DIR"
      ./util/tracer_nvbit/install_nvbit.sh
      ./get-accel-sim-traces.py
      exit 0
    }
  tar xzf traces.tgz
  rm traces.tgz
fi

# 3) Ejecutar la simulacion sobre la traza descargada
cd "$ACCEL_DIR"
source ./gpu-simulator/setup_environment.sh

echo ""
echo "==> Ejecutando simulacion Accel-Sim sobre vectoradd (Tesla V100)..."
echo ""

./gpu-simulator/bin/release/accel-sim.out \
  -trace "$TRACES_DIR/traces/kernelslist.g" \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config \
  2>&1 | tail -60

echo ""
echo "==> Listo. Las stats finales muestran ciclos, IPC y cache breakdown."
echo "    Comparen con los resultados del ejercicio 01 (GPGPU-Sim execution-driven)"
echo "    sobre el mismo kernel."
