#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

static void skipComments(std::istream& in) {
    char c;
    while (in >> std::ws && in.peek() == '#') {
        std::string line;
        std::getline(in, line);
    }
}

static Image readPgm(const fs::path& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open " + filename.string());
    }

    std::string magic;
    in >> magic;
    if (magic != "P5") {
        throw std::runtime_error("Only binary PGM (P5) files are supported: " +
                                 filename.string());
    }

    skipComments(in);
    int width, height, maxValue;
    in >> width;
    skipComments(in);
    in >> height;
    skipComments(in);
    in >> maxValue;
    in.get();  // Consume the single whitespace byte before pixel data.

    if (width <= 0 || height <= 0 || maxValue != 255) {
        throw std::runtime_error("Invalid PGM header: " + filename.string());
    }

    Image image{width, height, std::vector<unsigned char>(
                                  static_cast<size_t>(width) * height)};
    in.read(reinterpret_cast<char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));

    if (in.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("Incomplete PGM image: " + filename.string());
    }

    return image;
}

static void writePgm(const fs::path& filename, const Image& image) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot create " + filename.string());
    }

    out << "P5\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels.data()),
              static_cast<std::streamsize>(image.pixels.size()));
}

__global__ void sobelKernel(const unsigned char* input,
                            unsigned char* output,
                            int width,
                            int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    const int p = y * width + x;

    const int gx =
        -static_cast<int>(input[p - width - 1]) +
        static_cast<int>(input[p - width + 1]) -
        2 * static_cast<int>(input[p - 1]) +
        2 * static_cast<int>(input[p + 1]) -
        static_cast<int>(input[p + width - 1]) +
        static_cast<int>(input[p + width + 1]);

    const int gy =
        -static_cast<int>(input[p - width - 1]) -
        2 * static_cast<int>(input[p - width]) -
        static_cast<int>(input[p - width + 1]) +
        static_cast<int>(input[p + width - 1]) +
        2 * static_cast<int>(input[p + width]) +
        static_cast<int>(input[p + width + 1]);

    const int magnitude = min(255, abs(gx) + abs(gy));
    output[p] = static_cast<unsigned char>(magnitude);
}

static void printUsage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --input <dir> --output <dir> [--block-x <n>] [--block-y <n>]\n"
        << "Example: " << program
        << " --input data/input --output data/output --block-x 16 --block-y 16\n";
}

int main(int argc, char** argv) {
    std::string inputDir;
    std::string outputDir;
    int blockX = 16;
    int blockY = 16;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            inputDir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--block-x" && i + 1 < argc) {
            blockX = std::stoi(argv[++i]);
        } else if (arg == "--block-y" && i + 1 < argc) {
            blockY = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (inputDir.empty() || outputDir.empty() || blockX <= 0 || blockY <= 0) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    fs::create_directories(outputDir);

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No .pgm files found in " << inputDir << "\n";
        return EXIT_FAILURE;
    }

    int processed = 0;
    size_t totalPixels = 0;
    double totalMs = 0.0;

    std::cout << "CUDA Sobel image processing\n";
    std::cout << "Input directory: " << inputDir << "\n";
    std::cout << "Output directory: " << outputDir << "\n";
    std::cout << "Images found: " << files.size() << "\n";
    std::cout << "Block size: " << blockX << " x " << blockY << "\n";

    for (const fs::path& file : files) {
        try {
            Image image = readPgm(file);
            Image result{image.width, image.height,
                         std::vector<unsigned char>(
                             static_cast<size_t>(image.width) * image.height)};

            const size_t bytes = image.pixels.size() * sizeof(unsigned char);
            unsigned char* d_input = nullptr;
            unsigned char* d_output = nullptr;

            CUDA_CHECK(cudaMalloc(&d_input, bytes));
            CUDA_CHECK(cudaMalloc(&d_output, bytes));
            CUDA_CHECK(cudaMemcpy(d_input, image.pixels.data(), bytes,
                                  cudaMemcpyHostToDevice));

            dim3 block(static_cast<unsigned int>(blockX),
                       static_cast<unsigned int>(blockY));
            dim3 grid(
                static_cast<unsigned int>((image.width + blockX - 1) / blockX),
                static_cast<unsigned int>((image.height + blockY - 1) / blockY));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));

            sobelKernel<<<grid, block>>>(
                d_input, d_output, image.width, image.height);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

            CUDA_CHECK(cudaMemcpy(result.pixels.data(), d_output, bytes,
                                  cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_output));

            const fs::path outputFile =
                fs::path(outputDir) / file.filename();
            writePgm(outputFile, result);

            ++processed;
            totalPixels += result.pixels.size();
            totalMs += milliseconds;

            std::cout << "Processed " << file.filename().string()
                      << " (" << image.width << "x" << image.height
                      << "), kernel time: " << milliseconds << " ms\n";
        } catch (const std::exception& ex) {
            std::cerr << "Error processing " << file << ": "
                      << ex.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "\nCompleted successfully.\n";
    std::cout << "Images processed: " << processed << "\n";
    std::cout << "Total pixels: " << totalPixels << "\n";
    std::cout << "Total GPU kernel time: " << totalMs << " ms\n";
    std::cout << "Average GPU kernel time: "
              << (totalMs / processed) << " ms/image\n";

    return EXIT_SUCCESS;
}
