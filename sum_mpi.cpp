#include <mpi.h>
#include <array>
#include <iostream>
#include <vector>

std::array<std::size_t, 2> SplitArray(std::size_t part, std::size_t parts, std::size_t length)
{
    std::size_t start = part * length / parts;
    std::size_t end = (part + 1) * length / parts;
    return { start, end };
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::size_t arrayLength = 1000000000;

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int p = 0; p < size; p++)
    {
        auto bounds = SplitArray(p, size, arrayLength);
        counts[p] = static_cast<int>(bounds[1] - bounds[0]);
        displs[p] = static_cast<int>(bounds[0]);
    }

    std::vector<long long> array;
    if (rank == 0)
    {
        array.assign(arrayLength, 1);
    }

    // =========================
    // MPI
    // =========================

    std::vector<long long> localPart(counts[rank]);
    MPI_Scatterv(array.data(), counts.data(), displs.data(), MPI_LONG_LONG,
        localPart.data(), counts[rank], MPI_LONG_LONG,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    long long localSum = 0;
    for (std::size_t i = 0; i < localPart.size(); i++)
    {
        localSum += localPart[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double mpiTime = (MPI_Wtime() - start) * 1000.0;

    long long mpiTotal = 0;
    MPI_Reduce(&localSum, &mpiTotal, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // =========================
    // RESULTS
    // =========================
    if (rank == 0)
    {
        std::cout << "=== MPI ===" << std::endl;
        std::cout << "Sum: " << mpiTotal << std::endl;
        std::cout << "Time: " << mpiTime << " ms\n" << std::endl;

        if (mpiTotal != static_cast<long long>(arrayLength))
        {
            std::cout << "ERROR: wrong sum!" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}