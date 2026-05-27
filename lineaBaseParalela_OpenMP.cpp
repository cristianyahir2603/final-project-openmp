#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

void generateMandelbrotOMP(std::vector<int>& image) {
    // Paralelización base con el planificador por defecto
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double zr = 0.0, zi = 0.0;
            double cr = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double ci = (y - HEIGHT / 2.0) * 4.0 / WIDTH;
            int iter = 0;
            
            while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
                double temp = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;
                iter++;
            }
            image[y * WIDTH + x] = iter;
        }
    }
}

void applyGaussianOMP(const std::vector<int>& input, std::vector<int>& output) {
    float kernel[5][5] = {
        {1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f},
        {4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f},
        {6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f},
        {4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f},
        {1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f}
    };

    // Paralelización base para la convolución
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 2; y < HEIGHT - 2; ++y) {
        for (int x = 2; x < WIDTH - 2; ++x) {
            float sum = 0.0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    sum += input[(y + ky) * WIDTH + (x + kx)] * kernel[ky + 2][kx + 2];
                }
            }
            output[y * WIDTH + x] = static_cast<int>(sum);
        }
    }
}

int main() {
    std::vector<int> image(WIDTH * HEIGHT, 0);
    std::vector<int> blurredImage(WIDTH * HEIGHT, 0);

    std::cout << "Iniciando Tarea A (Mandelbrot OpenMP)..." << std::endl;
    auto startA = std::chrono::high_resolution_clock::now();
    generateMandelbrotOMP(image);
    auto endA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diffA = endA - startA;
    std::cout << "Tiempo Tarea A: " << diffA.count() << " s\n";

    std::cout << "Iniciando Tarea B (Convolución OpenMP)..." << std::endl;
    auto startB = std::chrono::high_resolution_clock::now();
    applyGaussianOMP(image, blurredImage);
    auto endB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diffB = endB - startB;
    std::cout << "Tiempo Tarea B: " << diffB.count() << " s\n";

    return 0;
}