// Generate 3D test input file for heat equation solver

#include <iostream>
#include <fstream>
#include <cmath>

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <nx> <ny> <nz> <output_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " 100 100 100 sphere_3d.dat" << std::endl;
        return 1;
    }

    int nx = std::atoi(argv[1]);
    int ny = std::atoi(argv[2]);
    int nz = std::atoi(argv[3]);
    std::string filename = argv[4];

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
