#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  long N = 100;
  long Nsize = 2000000/sizeof(int); // int is 4 Bytes
  double tt = MPI_Wtime(); 
//   int token = 0;
  int* msg = (int*) malloc(Nsize*sizeof(int));
  for(int repeat = 0; repeat < N; repeat++){
  
        // Receive from the lower process and send to the higher process. Take care
        // of the special case when you are the first process to prevent deadlock.
        if (world_rank != 0) {
            MPI_Recv(msg, Nsize, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
        } else {
            // Set the token's value if you are process 0
            for (long i = 0; i < Nsize; i++) msg[i] = 42;
        }
        MPI_Send(msg, Nsize, MPI_INT, (world_rank + 1) % world_size, 0,
                MPI_COMM_WORLD);
        // Now process 0 can receive from the last process. This makes sure that at
        // least one MPI_Send is initialized before all MPI_Recvs (again, to prevent
        // deadlock)
        if (world_rank == 0) {
            MPI_Recv(msg, Nsize, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
  }
   tt = MPI_Wtime() - tt;
//   if (!world_rank) printf("Total sum for %d process is %d\n", world_size, token);
  if (!world_rank) printf("ring latency: %e ms\n", 1000*tt/(world_size*N));
  if (!world_rank) printf("ring bandwidth: %e GB/s\n", (world_size*N*sizeof(int)*Nsize)/tt/1e9);

  MPI_Finalize();
}