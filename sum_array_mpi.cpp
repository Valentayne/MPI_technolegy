#include <mpi.h>
#include <array>
#include <iostream>
#include <vector>

const int N = 1000000000;

std::array<std::size_t, 2> SplitArray(std::size_t part, std::size_t parts, std::size_t length)
{
    std::size_t start = part * length / parts;
    std::size_t end = (part + 1) * length / parts;
    return { start, end };
}

int Waves(std::vector<int>& arr)
{
    if (arr.empty())
    {
        return 0;
    }

    int n = static_cast<int>(arr.size());  
    while (n > 1)                           
    {
        int pairs = n / 2;
        for (int i = 0; i < pairs; i++)
        {
            int j = n - 1 - i;              
            arr[i] += arr[j];               
        }
        n -= pairs;                       
    }
    return arr[0];
}

void Worker()
{
    int range[2];
    MPI_Recv(range, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int pairs = range[1] - range[0];

    std::vector<int> local(2 * pairs, 1);

    MPI_Barrier(MPI_COMM_WORLD);           

    int sum = Waves(local);
    MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void Controller(int workers)
{
    for (int w = 1; w <= workers; w++)
    {
        auto bounds = SplitArray(w - 1, workers, N / 2);
        int range[2] = { static_cast<int>(bounds[0]), static_cast<int>(bounds[1]) };
        MPI_Send(range, 2, MPI_INT, w, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);  
    double start = MPI_Wtime();

    std::vector<int> sums(workers);
    for (int w = 1; w <= workers; w++)
    {
        MPI_Recv(&sums[w - 1], 1, MPI_INT, w, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int total = Waves(sums);

    double time = (MPI_Wtime() - start) * 1000.0;

    std::cout << "Elements: " << N << ", workers: " << workers << " + controller" << std::endl;
    std::cout << "Sum: " << total << " (" << (total == N ? "correct" : "ERROR") << ")" << std::endl;
    std::cout << "Time: " << time << " ms" << std::endl;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        Controller(size - 1);
    }
    else
    {
        Worker();
    }

    MPI_Finalize();
    return 0;
}