CXX := nvcc
TARGET := image_processing.exe
SOURCE := image_processing.cu

NVCC_FLAGS := -std=c++17 -O2

build:
	$(CXX) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf data/output
	rm -f execution.log

run: build
	./$(TARGET) --input data/input --output data/output --block-x 16 --block-y 16

.PHONY: build clean run
