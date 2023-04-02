#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {

    int p = 0, chunk_size = 0;
    #pragma omp parallel shared(p, chunk_size)
    {
        p = omp_get_num_threads();
        chunk_size = n/p;
        
        // Partial sum calculation
        int t = omp_get_thread_num(); 
        prefix_sum[t*chunk_size] = A[t*chunk_size];
        for(long j = t*chunk_size+1; j < t*chunk_size + chunk_size; j++){prefix_sum[j] = prefix_sum[j-1] + A[j];} 
    }
    // Residual calculation in serial vector
    long *resid = new long[p-1]{0};
    long k = 1;
    resid[0] = prefix_sum[chunk_size-1];
    for(long i = 2*chunk_size-1; i < n - chunk_size; i+=chunk_size){resid[k] = prefix_sum[i] + resid[k-1]; k = k+1;}

    // adding residuals to the prefix values
    #pragma omp parallel shared(p, chunk_size, resid)
    {
      int t = omp_get_thread_num();
      if(t < p-1)
      {
          for (long j = (t + 1)*chunk_size; j < (t + 2)*chunk_size; j++ ){prefix_sum[j] += resid[t];}
      }
    }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;
  for (long i = 0; i < N; i++) B0[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  int ps[7] = {2,4,8,16,32,64,128};
  for(auto p : ps)
  {
    omp_set_num_threads(p);
    tt = omp_get_wtime();
    scan_omp(B1, A, N);
    printf("parallel-scan   = %fs for %d threads,  ", omp_get_wtime() - tt, p);
    long err = 0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);
  }
  free(A);
  free(B0);
  free(B1);
  return 0;
}
