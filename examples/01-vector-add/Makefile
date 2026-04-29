# Makefile para correr CUDA bajo GPGPU-Sim
#
# El truco: GPGPU-Sim provee su propia libcudart.so. Cuando linkeamos contra
# ella y ejecutamos el binario, las llamadas a la API CUDA se interceptan y
# se simulan ciclo a ciclo en lugar de ir al driver de NVIDIA.
#
# La config de la GPU simulada se elige copiando un gpgpusim.config al cwd
# antes de ejecutar. Aca usamos QV100 (Volta) por defecto.

NVCC = nvcc
TARGET = vector_add
SRC = vector_add.cu

# -cudart shared : linkear dinamicamente -> GPGPU-Sim puede interceptar
# -O3            : optimizacion estandar
# --compiler-options -fPIC : requerido por GPGPU-Sim
NVCCFLAGS = -O3 --cudart shared --compiler-options -fPIC

# Config de GPU a simular. Otras opciones que vienen en el repo:
#   SM7_QV100   = Tesla V100 (Volta)      <- default, moderna y educativa
#   SM75_RTX2060 = RTX 2060 (Turing)
#   SM6_TITANX  = TITAN X (Pascal)
GPU_CONFIG = SM7_QV100

# Path donde GPGPU-Sim deja las configs de GPU
GPGPUSIM_CONFIGS = /opt/gpgpu-sim_distribution/configs/tested-cfgs

.PHONY: all run clean config

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Copia la config de la GPU simulada al cwd (GPGPU-Sim la lee de ./gpgpusim.config)
config:
	cp $(GPGPUSIM_CONFIGS)/$(GPU_CONFIG)/gpgpusim.config .
	cp $(GPGPUSIM_CONFIGS)/$(GPU_CONFIG)/config_volta_islip.icnt . 2>/dev/null || true
	cp $(GPGPUSIM_CONFIGS)/$(GPU_CONFIG)/*.icnt . 2>/dev/null || true

run: $(TARGET) config
	@echo "=== Simulando en GPU $(GPU_CONFIG) ==="
	./$(TARGET)

clean:
	rm -f $(TARGET) gpgpusim.config *.icnt _cuobjdump_* _ptx_jit_* gpgpu_inst_stats.txt
