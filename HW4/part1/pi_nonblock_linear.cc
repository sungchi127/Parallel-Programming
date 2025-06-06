#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	unsigned int seed = time(NULL)*world_rank;
	long long int total, work_cnt, n;
	total = 0;
	work_cnt = 0;
	n = tosses/world_size;
	float x, y, z;

	MPI_Barrier(MPI_COMM_WORLD);
	while(n--){
		x = rand_r(&seed)/((float)RAND_MAX);
		y = rand_r(&seed)/((float)RAND_MAX);
		z = x*x+y*y;
		if(z <= 1.0) ++work_cnt;
	}	
	MPI_Request requests;

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Isend(&work_cnt, 1, MPI_LONG_LONG_INT, 0, world_rank, MPI_COMM_WORLD, &requests);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        // MPI_Request requests[];
		MPI_Status status;
		total = work_cnt;
		for(int source = 1 ; source < world_size ; source++){
			MPI_Irecv(&work_cnt, 1, MPI_LONG_LONG_INT, source, source, MPI_COMM_WORLD, &requests);
			MPI_Wait(&requests, &status);
			total += work_cnt;
		}
        // MPI_Waitall();
    }

    if (world_rank == 0)
    {
        // TODO: PI result
		double cn = (double)total/tosses;
		pi_result = 4.0*cn;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
