# CUDA 관련 설정
CUDA_ROOT_DIR := /usr/local/cuda-11.7
CUDA_LIB_DIR := -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR := -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS := -lcudart

# 컴파일러 설정
CC := /usr/bin/gcc
NVCC := nvcc
RM := /bin/rm

CFLAGS := -Wall -Wextra -O3 -fomit-frame-pointer -march=native -mavx2 -mfma -fPIC 
NVCC_FLAGS := -arch=sm_80 -Xcompiler -fPIC  -ccbin $(CC) $(CUDA_INC_DIR)

# 디렉토리
SRC_DIR := src
OBJ_DIR := bin

# 소스 파일 목록
C_SOURCES := $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES := $(wildcard $(SRC_DIR)/*_gpu.cu)

# 오브젝트 파일 목록
C_OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(C_SOURCES))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))

TARGET_SO := caltest.so

# 기본 규칙
all: $(TARGET_SO)

# OBJ 디렉토리 생성
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# C 파일 컴파일
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA 파일 컴파일
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 모든 오브젝트 파일을 링크해 .so 생성
$(TARGET_SO): $(C_OBJS) $(CU_OBJS)
	$(CC) -shared -o $@ $^ -lgmp -lm $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# 테스트 실행 파일 (선택)
test/%: test/%.c $(C_SOURCES)
	$(CC) -Wall -Wextra -O2 -o $@ $(C_SOURCES) $< -lgmp

# 정리
clean:
	$(RM) -rf $(OBJ_DIR)/*.o $(TARGET_SO)
