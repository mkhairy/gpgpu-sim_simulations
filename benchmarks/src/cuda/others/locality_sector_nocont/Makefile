include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

#INCLUDE = /usr/local/cuda/include:../common/inc/
LIB := /usr/local/cuda/lib:../common/lib/

INCLUDE := ../common/inc/:$(CUDA_DIR)/include
LIB := ../common/lib/:$(CUDA_LIB_DIR) 

LIBRARIES = cudart.lib cutil32.lib

SRC = locality_sector_nocont.cu

EXE = locality_sector_nocont

# -Xptxas -dlcm=cg
#-ptxas-options=-v

release:
	$(CC)  -arch=sm_61  -Xptxas -v -Xptxas -dlcm=cg  $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debug:
	$(CC) -arch=sm_20 -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 
