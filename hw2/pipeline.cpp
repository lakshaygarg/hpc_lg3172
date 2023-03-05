#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <vector>
using namespace std;

void inner_prod_2(double *res, double *a, double *b, int N, string type)
{
    if(type == "unroll_0")
    {
        for(long i = 0; i < N; i++)
        {
            *res += a[i]*b[i];
        }
    }
    else if(type == "unroll_1")
    {
       double sum1 = 0, sum2 = 0;
        for(long i = 0; i < N/2 - 1; i++)
        {
            sum1 += a[2*i] * b[2*i];
            sum2 += a[2*i + 1] * b[2*i + 1];
        }
        *res = sum1+sum2;     
    }
    else if(type == "unroll_2")
    {
        double sum1 = 0, sum2 = 0;
        for (long i = 0; i < N/2-1; i ++) {
            sum1 += *(a + 0) * *(b + 0);
            sum2 += *(a + 1) * *(b + 1);
            a += 2; b += 2;
        }
        *res = sum1+sum2;   
    }
    else if(type == "unroll_3")
    {
        double sum1 = 0, sum2 = 0, temp1 = 0, temp2 = 0;
        for (long i = 0; i < N/2-1; i ++) {
            temp1 = *(a + 0) * *(b + 0);
            temp2 = *(a + 1) * *(b + 1);
            sum1 += temp1; sum2 += temp2;
            a += 2; b += 2;
        }
        *res = sum1+sum2;   
    }

    else if(type == "unroll_4")
    {
        double sum1 = 0, sum2 = 0, temp1 = 0, temp2 = 0;
        for (long i = 0; i < N/2-1; i ++) {
            sum1 += temp1;
            temp1 = *(a + 0) * *(b + 0);
            sum2 += temp2;
            temp2 = *(a + 1) * *(b + 1);
            a += 2; b += 2;
        }
        *res = sum1+sum2;
    }
}

void inner_prod_4(double *res, double *a, double *b, int N, string type)
{
    if(type == "unroll_0")
    {
        for(long i = 0; i < N; i++)
        {
            *res += a[i]*b[i];
        }
    }
    else if(type == "unroll_1")
    {
       double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        for(long i = 0; i < N/4 - 1; i++)
        {
            sum1 += a[4*i] * b[4*i];
            sum2 += a[4*i + 1] * b[4*i + 1];
            sum3 += a[4*i + 2] * b[4*i + 2];
            sum4 += a[4*i + 3] * b[4*i + 3];
        }
        *res = sum1+sum2+sum3+sum4;     
    }
    else if(type == "unroll_2")
    {
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        for (long i = 0; i < N/4 - 1; i ++) {
            sum1 += *(a + 0) * *(b + 0);
            sum2 += *(a + 1) * *(b + 1);
            sum3 += *(a + 2) * *(b + 2);
            sum2 += *(a + 3) * *(b + 3);
            a += 4; b += 4;
        }
        *res = sum1+sum2+sum3+sum4;   
    }
    else if(type == "unroll_3")
    {
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, temp1 = 0,temp3 = 0,temp4 = 0, temp2 = 0;
        for (long i = 0; i < N/4 - 1; i ++) {
            temp1 = *(a + 0) * *(b + 0);
            temp2 = *(a + 1) * *(b + 1);
            temp3 = *(a + 2) * *(b + 2);
            temp4 = *(a + 3) * *(b + 3);
            sum1 += temp1; sum2 += temp2;
            sum3 += temp3; sum4 += temp4;
            a += 4; b += 4;
        }
        *res = sum1+sum2+sum3+sum4; 
    }

    else if(type == "unroll_4")
    {
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, temp1 = 0,temp3 = 0,temp4 = 0, temp2 = 0;
        for (long i = 0; i < N/4 - 1; i ++) {
            sum1 += temp1;
            temp1 = *(a + 0) * *(b + 0);
            sum2 += temp2;
            temp2 = *(a + 1) * *(b + 1);
            sum3 += temp3;
            temp3 = *(a + 2) * *(b + 2);
            sum4 += temp4;
            temp4 = *(a + 3) * *(b + 3);
            a += 4; b += 4;
        }
            *res = sum1+sum2+sum3+sum4; 
    }
}


int main()
{
  Timer t;
  long NREPEATS = 10;
  long n = 100000000;
  double* a = (double*) malloc(n * sizeof(double)) ;
  double* b = (double*) malloc(n * sizeof(double)) ;
  double* res = (double*) malloc(sizeof(double));
  for (long i = 0; i < n; i++) {a[i] = drand48();}
  for (long i = 0; i < n; i++) {b[i] = drand48();}

  vector<string> types = {"unroll_0", "unroll_1","unroll_2","unroll_3","unroll_4"};
  for(string type : types){  
        t.tic();
        for(long i = 0; i < NREPEATS; i++)
        {
            *res = 0;
            inner_prod_2(res,a,b,n,type);
        }
        printf("Run time for %s for factor of 2 is %10f\n", type.c_str(), t.toc());
    }

   for(string type : types) {
        t.tic();
        for(long i = 0; i < NREPEATS; i++)
        {
            *res = 0;
            inner_prod_4(res,a,b,n,type);
        }
        printf("Run time for %s for factor of 4 is %10f\n", type.c_str(), t.toc());
    }
    free(a);free(b);free(res);
    return 0;
}
