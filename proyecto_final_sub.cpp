#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

// --- TAREA A: Generación con schedule(runtime) para pruebas empíricas ---
void generateMandelbrotRuntime(std::vector<int>& image) {
    // schedule(runtime) lee la variable de entorno OMP_SCHEDULE al ejecutar
    #pragma omp parallel for schedule(runtime) collapse(2)
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

// --- TAREA B: Convolución con Vectorización (SPMD) ---
void applyGaussianOMP_SIMD(const std::vector<int>& input, std::vector<int>& output) {
    float kernel[5][5] = {
        {1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f},
        {4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f},
        {6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f},
        {4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f},
        {1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f}
    };

    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 2; y < HEIGHT - 2; ++y) {
        for (int x = 2; x < WIDTH - 2; ++x) {
            float sum = 0.0;
            for (int ky = -2; ky <= 2; ++ky) {
                // Forzamos la vectorización en el bucle más interno
                #pragma omp simd reduction(+:sum)
                for (int kx = -2; kx <= 2; ++kx) {
                    sum += input[(y + ky) * WIDTH + (x + kx)] * kernel[ky + 2][kx + 2];
                }
            }
            output[y * WIDTH + x] = static_cast<int>(sum);
        }
    }
}

// --- HISTOGRAMA 1: Exclusión Mutua (Atomic) ---
void calcHistogramAtomic(const std::vector<int>& image, int hist[257]) {
    #pragma omp parallel for
    for (size_t i = 0; i < image.size(); ++i) {
        #pragma omp atomic
        hist[image[i]]++;
    }
}

// --- HISTOGRAMA 2: Cláusula Reduction ---
void calcHistogramReduction(const std::vector<int>& image, int hist[257]) {
    // OpenMP 4.5+ soporta reducción en arreglos directamente
    #pragma omp parallel for reduction(+:hist[:257])
    for (size_t i = 0; i < image.size(); ++i) {
        hist[image[i]]++;
    }
}

int main() {
    std::vector<int> image(WIDTH * HEIGHT, 0);
    std::vector<int> blurredImage(WIDTH * HEIGHT, 0);
    int histAtomic[257] = {0};
    int histReduction[257] = {0};

    // 1. Tarea A
    std::cout << "Iniciando Tarea A (Mandelbrot Runtime)..." << std::endl;
    auto startA = std::chrono::high_resolution_clock::now();
    generateMandelbrotRuntime(image);
    auto endA = std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo Tarea A: " << std::chrono::duration<double>(endA - startA).count() << " s\n";

    // 2. Tarea B
    std::cout << "Iniciando Tarea B (Convolución)..." << std::endl;
    auto startB = std::chrono::high_resolution_clock::now();
    applyGaussianOMP_SIMD(image, blurredImage);
    auto endB = std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo Tarea B: " << std::chrono::duration<double>(endB - startB).count() << " s\n";

    // 3. Sincronización: Atomic vs Reduction
    std::cout << "Calculando Histograma (Atomic)..." << std::endl;
    auto startH1 = std::chrono::high_resolution_clock::now();
    calcHistogramAtomic(blurredImage, histAtomic);
    auto endH1 = std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo Histograma Atomic: " << std::chrono::duration<double>(endH1 - startH1).count() << " s\n";

    std::cout << "Calculando Histograma (Reduction)..." << std::endl;
    auto startH2 = std::chrono::high_resolution_clock::now();
    calcHistogramReduction(blurredImage, histReduction);
    auto endH2 = std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo Histograma Reduction: " << std::chrono::duration<double>(endH2 - startH2).count() << " s\n";

    return 0;
}