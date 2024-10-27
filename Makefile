CUDA_C = nvcc
CUDA_FLAGS = -O0

SRC_FOLDER = src
KERNELS_FOLDER = $(SRC_FOLDER)/kernels
BUILD_FOLDER = build
BUILD_KERNELS_FOLDER = $(BUILD_FOLDER)/kernels

CUDA_SRC = $(wildcard $(SRC_FOLDER)/*.cu) $(wildcard $(KERNELS_FOLDER)/*.cu)
CXX_SRC = $(wildcard $(SRC_FOLDER)/*.cpp)

OBJ = $(patsubst $(SRC_FOLDER)/%.cu, $(BUILD_FOLDER)/%.o, $(CUDA_SRC)) \
	  $(patsubst $(SRC_FOLDER)/%.cpp, $(BUILD_FOLDER)/%.o, $(CXX_SRC))

TARGET = $(BUILD_FOLDER)/kernels.out

all: $(BUILD_FOLDER) $(BUILD_KERNELS_FOLDER) $(TARGET)

$(BUILD_FOLDER):
	mkdir -p $(BUILD_FOLDER)

$(BUILD_KERNELS_FOLDER):
	mkdir -p $(BUILD_KERNELS_FOLDER)

$(TARGET): $(OBJ)
	$(CUDA_C) $(CUDA_FLAGS) -o $@ $^

$(BUILD_FOLDER)/%.o: $(SRC_FOLDER)/%.cu | $(BUILD_FOLDER)
	$(CUDA_C) $(CUDA_FLAGS) -c $< -o $@

$(BUILD_KERNELS_FOLDER)/%.o: $(KERNELS_FOLDER)/%.cu | $(BUILD_KERNELS_FOLDER)
	$(CUDA_C) $(CUDA_FLAGS) -c $< -o $@

$(BUILD_FOLDER)/%.o: $(SRC_FOLDER)/%.cpp | $(BUILD_FOLDER)
	$(CUDA_C) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_FOLDER)
