INC=-I/usr/local/cuda/include
NVCC=nvcc
NVCC_OPT=-g -G
CPU_OPT=-std=c++2a -pg
FILE_PROF= prof/image.cpp prof/image_io.cpp


all:
	@echo "Choose cpu, gpu or opti"

opti:
	$(NVCC) $(NVCC_OPT) gpu_opti.cu $(FILE_PROF) -o gpu_opti
	./gpu_opti

gpu:
	$(NVCC) $(NVCC_OPT) gpu.cu $(FILE_PROF) -o gpu
	./gpu

cpu:
	g++ $(CPU_OPT) cpu.cpp $(FILE_PROF) -o cpu
	./cpu

clean:
	rm -f gpu
	rm -f gpu_opti
	rm -f cpu
