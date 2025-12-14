// Generate 3D test input file for heat equation solver

#include <iostream>
#include <fstream>
#include <cmath>

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 5) {
        std::cout << "Usage: " << argv[0] << " <output_file> [nx] [ny] [nz]" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " sphere.dat              # Uses default 100x100x100" << std::endl;
        std::cout << "  " << argv[0] << " sphere.dat 200          # Creates 200x200x200" << std::endl;
        std::cout << "  " << argv[0] << " sphere.dat 200 150 100  # Creates 200x150x100" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // Default or user-specified dimensions
    int nx = 100;
    int ny = 100;
    int nz = 100;

    if (argc >= 3) nx = std::atoi(argv[2]);
    if (argc >= 4) ny = std::atoi(argv[3]);
    if (argc >= 5) nz = std::atoi(argv[4]);

    // If only one dimension given, make it cubic
    if (argc == 3) {
        ny = nx;
        nz = nx;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    // Write header
    file << "# " << nx << " " << ny << " " << nz << std::endl;

    // Generate 3D temperature field with a hot sphere in the center
    double cx = nx / 2.0;
    double cy = ny / 2.0;
    double cz = nz / 2.0;
    double radius = std::min(std::min(nx, ny), nz) / 6.0;

    std::cout << "Generating 3D field: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Hot sphere at center (" << cx << ", " << cy << ", " << cz << ") with radius " << radius << std::endl;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double dx = i - cx;
                double dy = j - cy;
                double dz = k - cz;
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                double temp;
                if (dist < radius) {
                    // Hot sphere in the center
                    temp = 95.0;
                } else {
                    // Cool surrounding
                    temp = 15.0;
                }

                file << temp << "\n";
            }
        }
        // Progress indicator
        if ((i + 1) % (nx / 10) == 0) {
            std::cout << "Progress: " << (100 * (i + 1) / nx) << "%" << std::endl;
        }
    }

    file.close();
    std::cout << "Successfully wrote 3D input file: " << filename << std::endl;
    std::cout << "Total points: " << (nx * ny * nz) << std::endl;

    return 0;
}
