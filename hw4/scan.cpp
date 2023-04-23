#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long N = 100000000;
    long *A;
    long *prefix;
    long *ref;
    if(world_rank == 0)
    {
        A = (long*) malloc(N * sizeof(long));
        for (long i = 0; i < N; i++) A[i] = rand();
        ref = (long*) malloc(N*sizeof(long));
        ref[0] = A[0];
        for(int i = 1; i <N ;i++){ref[i] = ref[i-1] + A[i];}
        prefix = (long*) malloc(N * sizeof(long));
    }
    double tt = MPI_Wtime(); 
    long *localP = (long*) malloc(N * sizeof(long)/ world_size);
    long *localA = (long*) malloc(N * sizeof(long)/ world_size);
    MPI_Scatter(A,N/world_size,MPI_LONG,localA,N/world_size, MPI_LONG,0,MPI_COMM_WORLD);

    localP[0] = localA[0];
    for(int i = 1; i < N/world_size; i++)
    {
        localP[i] = localP[i-1] + localA[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    long localOff = localP[N/world_size - 1];
    long *offsets = (long*) malloc(sizeof(long) * world_size);
    MPI_Allgather(&localOff, 1, MPI_LONG, offsets, 1, MPI_LONG, MPI_COMM_WORLD);

    long loff = 0;
    for(int i = 0; i < world_rank; i++){loff += offsets[i];}
    for(int i = 0; i < N/world_size; i++)
    {
        localP[i] += loff;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(localP,N/world_size,MPI_LONG,prefix,N/world_size,MPI_LONG,0,MPI_COMM_WORLD);

    tt = MPI_Wtime() - tt;
    if (!world_rank)
    {
        long error = 0;
        for(int i = 0; i  <N; i++){error += ref[i] - prefix[i];}
        printf("N = %ld, Error = %ld, Wtime = %f ms\n", N,error,tt * 1000);
        free(A);free(prefix);free(ref);
    }
    free(localP);free(localA);free(offsets);
    MPI_Finalize();
}