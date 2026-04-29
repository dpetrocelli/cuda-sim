NVCC = nvcc
TARGET = matrix_mul
SRC = matrix_mul.cu
NVCCFLAGS = -O3 --cudart shared --compiler-options -fPIC
GPU_CONFIG = SM7_QV100
GPGPUSIM_CONFIGS = /opt/gpgpu-sim_distribution/configs/tested-cfgs

.PHONY: all run clean config

all: $(TARGET)
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

config:
	cp $(GPGPUSIM_CONFIGS)/$(GPU_CONFIG)/gpgpusim.config .
	cp $(GPGPUSIM_CONFIGS)/$(GPU_CONFIG)/*.icnt . 2>/dev/null || true

run: $(TARGET) config
	@echo "=== Simulando MATMUL ($(GPU_CONFIG)) ==="
	./$(TARGET)

clean:
	rm -f $(TARGET) gpgpusim.config *.icnt _cuobjdump_* _ptx_jit_* gpgpu_inst_stats.txt
