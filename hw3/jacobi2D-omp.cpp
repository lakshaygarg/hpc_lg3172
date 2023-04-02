#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#define MAX_ITER 10000
using namespace std;
#if defined(_OPENMP)
#include <omp.h>
#else
#include <chrono>
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif
#include <chrono>

// jacobi implementation
void jacobi(int N, double &res)
{
    double *u_old = new double[(N+2)*(N+2)]{0.0};
    double *u_new = new double[(N+2)*(N+2)]{0.0};
    double h2 = pow(1.0/(N + 1),2), RES = 1.0*N/1e4;
    res = 1e10;
    int iter = 0;
    while(iter <= MAX_ITER && res > RES)
    {
        res = 0.0;
        for(int i = 1; i <= N; i++)
        {
            for(int j = 1; j <= N; j++)
            {
                u_new[i + (N+2)*j] = (h2 + u_old[i-1 + (N+2)*j] + u_old[i + (N+2)*(j-1)] + u_old[i+1 + (N+2)*j] + u_old[i + (N+2)*(j+1)])/4.0;
            }
        }
        for(int i = 1; i <= N; i++)
        {
            for(int j = 1; j <= N; j++)
            {
                res += pow((-u_new[i-1 + (N+2)*j] - u_new[i + (N+2)*(j-1)] + 4*u_new[i + (N+2)*j] - u_new[i+1 + (N+2)*j] - u_new[i + (N+2)*(j+1)])/h2 - 1.0,2.0);
            }
        }
        res = pow(res,0.5);
        iter++;
        double *u_temp = u_old;
        u_old = u_new;
        u_new = u_temp;
    }
    delete[] u_old, u_new;
}

void jacobi_omp(int N, double &res)
{
    #if defined(_OPENMP)
    double *u_old = new double[(N+2)*(N+2)]{0.0};
    double *u_new = new double[(N+2)*(N+2)]{0.0};
    double h2 = pow(1.0/(N + 1),2), RES = 1.0*N/1e4;
    res = 1e10;
    int iter = 0;

    while(iter <= MAX_ITER && res > RES)
    {
        res = 0.0;
        #pragma omp parallel
        {
            #pragma omp for
            for(int i = 1; i <= N; i++)
            {
                for(int j = 1; j <= N; j++)
                {
                    u_new[i + (N+2)*j] = (h2 + u_old[i-1 + (N+2)*j] + u_old[i + (N+2)*(j-1)] + u_old[i+1 + (N+2)*j] + u_old[i + (N+2)*(j+1)])/4.0;
                }
            }

            #pragma omp for reduction(+ : res)
            for(int j = 1; j <= N; j++)
            {
                for(int i = 1; i <= N; i++)
                {
                    res += pow((-u_new[i-1 + (N+2)*j] - u_new[i + (N+2)*(j-1)] + 4*u_new[i + (N+2)*j] - u_new[i+1 + (N+2)*j] - u_new[i + (N+2)*(j+1)])/h2 - 1.0,2.0);
                }
            }
        }
        double *u_temp = u_old;
        u_old = u_new;
        u_new = u_temp;
        iter++;
        res = pow(res,0.5);
    }
    delete[] u_old, u_new;
    #else
    // jacobi(N, res);
    #endif
}

int main()
{
    int Ns[] = {50,200,500};
    double res = 0.0;
    for(auto N : Ns)
    {   
        #if defined(_OPENMP)
        double tt = omp_get_wtime();
        jacobi(N, res);
        printf("Serial Implementation runtime : %fs, points : %d, intial error : %f, final error : %f \n", omp_get_wtime() - tt, N*N, 1.0*N , res);
        #else
        auto start = std::chrono::high_resolution_clock::now();
        jacobi(N, res);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedSeconds = end - start;
        double seconds = elapsedSeconds.count();
        printf("Serial Implementation runtime : %fs, points : %d, intial error : %f, final error : %f \n", seconds, N*N, 1.0*N , res);
        #endif
    }
    int ps[] = {2,4,8,16};
    for(auto N : Ns)
    {
        for (auto p : ps)
        {
            #if defined(_OPENMP)
            omp_set_num_threads(p);
            double tt = omp_get_wtime();
            jacobi_omp(N, res);
            printf("Parallel Implementation runtime : %fs, Threads : %d, points : %d, intial error : %f, final error : %f \n", omp_get_wtime() - tt, p,N*N, 1.0*N , res);
            #else
            auto start = std::chrono::high_resolution_clock::now();
            jacobi(N, res);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedSeconds = end - start;
            double seconds = elapsedSeconds.count();
            printf("Parallel Implementation runtime : %fs, Threads : %d, points : %d, intial error : %f, final error : %f \n", seconds, p,N*N, 1.0*N , res);
            #endif
        }
    }
    return 0;
}