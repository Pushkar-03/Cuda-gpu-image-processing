# CUDA GPU Image Processing Project

## Project title

**GPU-Accelerated Sobel Edge Detection for Batch Image Processing**

## Description

This project implements image processing on the GPU using a custom CUDA kernel. It reads a large collection of grayscale PGM images, applies Sobel edge detection to every image, and writes the processed images to an output directory.

The Sobel operation calculates horizontal and vertical image gradients for each pixel. Each CUDA thread processes one image pixel, while a two-dimensional CUDA grid covers the complete image.

The project is designed to demonstrate GPU computation on a large amount of image data. The included data-generation script can create hundreds of small test images so the program can be tested with a workload representative of the assignment requirements.

## GPU computation

The main computation is performed by the `sobelKernel` CUDA kernel in `image_processing.cu`.

For each pixel, the GPU reads the surrounding 3x3 neighborhood, calculates the Sobel X and Y gradients, and produces an edge magnitude. The kernel is launched with a configurable 2D CUDA block size.

## Build

The project requires an NVIDIA GPU, CUDA toolkit, and `nvcc`.

```bash
make build
```

This creates:

```text
image_processing.exe
```

## Generate a large dataset

The repository includes a Python script that creates 200 small grayscale PGM images:

```bash
python3 generate_dataset.py --count 200 --width 256 --height 256
```

The images are written to:

```text
data/input/
```

## Run

```bash
./image_processing.exe --input data/input --output data/output --block-x 16 --block-y 16
```

Or:

```bash
make run
```

The program prints the number of images, image dimensions, CUDA block size, and GPU kernel execution time for every image.

## Try another CUDA configuration

For example:

```bash
./image_processing.exe --input data/input --output data/output --block-x 32 --block-y 8
```

The command-line arguments make it possible to experiment with different CUDA configurations.

## Output

Processed images are written to:

```text
data/output/
```

The output images contain the detected edges.

## Proof of execution

A sample execution can be captured with:

```bash
./image_processing.exe --input data/input --output data/output --block-x 16 --block-y 16 | tee execution.log
```

The resulting `execution.log` shows that the program successfully processed the complete image dataset and reports GPU kernel execution times.

For the final submission, include screenshots or logs showing:
- the build completed successfully,
- the program processed hundreds of images,
- output images were created,
- CUDA kernel execution times were reported.

## Project structure

```text
.
├── image_processing.cu
├── Makefile
├── README.md
├── generate_dataset.py
└── data/
    ├── input/
    └── output/
```
