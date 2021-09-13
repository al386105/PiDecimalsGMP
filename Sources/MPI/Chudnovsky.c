#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "../../Headers/Sequential/Chudnovsky.h"
#include "../../Headers/MPI/OperationsMPI.h"

#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Chudnovsky formula implementation                                                *
 * This version does not computes all the factorials                                *
 * It implements a sequential method and another one that can use multiple          *
 * processes and threads in hybrid way.                                             *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula:                                                              *
 *     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)             *
 *    --------------------  = SUMMATORY( ----------------------------- ),  n >=0    *
 *            pi                            (n!)^3 (3n)! (-640320)^3n               *
 *                                                                                  *
 * Some operands of the formula are coded as:                                       *
 *      dep_a_dividend = (6n)!                                                      *
 *      dep_a_divisor  = (n!)^3 (3n)!                                               *
 *      e              = 426880 sqrt(10005)                                         *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula dependencies:                                                 *
 *                     (6n)!         (12n + 10)(12n + 6)(12n + 2)                   *
 *      dep_a(n) = --------------- = ---------------------------- * dep_a(n-1)      *
 *                 ((n!)^3 (3n)!)              (n + 1)^3                            *
 *                                                                                  *
 *      dep_b(n) = (-640320)^3n = (-640320)^3(n-1) * (-640320)^3)                   *
 *                                                                                  *
 *      dep_c(n) = (545140134n + 13591409) = dep_c(n - 1) + 545140134               *
 *                                                                                  *
 ************************************************************************************/


/*
 * This method provides an optimal distribution for each thread of any proc
 * based on the Chudnovsky iterations analysis.
 * IMPORTANT: The number of threads used MUST be the same in every process
 * IMPORTANT: (num_procs * num_threads) % 4 == 0 OR (num_procs * num_threads) == 2
 * It returns an array of three integers:
 *   distribution[0] -> block size
 *   distribution[1] -> block start
 *   distribution[2] -> block end 
 */
int * get_distribution(int num_procs, int proc_id, int num_threads, int thread_id, int num_iterations){
    int * distribution, i, block_size, block_start, block_end, my_row, my_column;
    float my_working_ratio; 

    float working_ratios[64][17] = { 
        59.50,	35.00,	21.35,	15.75,	12.77,	10.85,	9.48,	8.46,	7.69,	7.06,	6.55,	6.11,	5.73,	5.41,	5.12,	4.87,	4.64,
        40.50,	24.50,	14.35,	10.50,	8.57,	7.22,	6.37,	5.66,	5.11,	4.68,	4.31,	4.00,	3.75,	3.52,	3.34,	3.19,	3.05,
        0.00,	21.00,	12.25,	9.45,	7.53,	6.34,	5.47,	4.90,	4.45,	4.12,	3.80,	3.54,	3.32,	3.14,	2.96,	2.80,	2.66,
        0.00,	19.50,	11.55,	8.40,	6.83,	5.87,	5.08,	4.52,	4.08,	3.70,	3.42,	3.23,	3.06,	2.87,	2.71,	2.57,	2.45,
        0.00,	0.00,	10.85,	7.75,	6.30,	5.40,	4.82,	4.26,	3.85,	3.51,	3.25,	3.00,	2.79,	2.63,	2.53,	2.43,	2.31,
        0.00,	0.00,	10.30,	7.65,	5.95,	5.05,	4.46,	4.09,	3.68,	3.35,	3.09,	2.88,	2.69,	2.54,	2.37,	2.23,	2.14,
        0.00,	0.00,	10.00,	7.35,	5.88,	4.80,	4.24,	3.79,	3.54,	3.25,	2.98,	2.77,	2.58,	2.44,	2.30,	2.19,	2.07,
        0.00,	0.00,	9.35,	7.00,	5.77,	4.68,	4.02,	3.65,	3.30,	3.10,	2.90,	2.67,	2.51,	2.35,	2.22,	2.11,	2.01,
        0.00,	0.00,	0.00,	6.75,	5.56,	4.68,	3.93,	3.46,	3.21,	2.93,	2.77,	2.61,	2.44,	2.28,	2.17,	2.04,	1.95,
        0.00,	0.00,	0.00,	6.65,	5.25,	4.61,	3.92,	3.40,	3.05,	2.87,	2.63,	2.50,	2.38,	2.25,	2.10,	2.00,	1.90,
        0.00,	0.00,	0.00,	6.55,	5.14,	4.48,	3.88,	3.34,	3.00,	2.72,	2.59,	2.39,	2.28,	2.18,	2.08,	1.95,	1.86,
        0.00,	0.00,	0.00,	6.20,	5.08,	4.23,	3.84,	3.37,	2.93,	2.67,	2.46,	2.36,	2.18,	2.09,	2.02,	1.93,	1.83,
        0.00,	0.00,	0.00,	0.00,	5.01,	4.14,	3.75,	3.31,	2.94,	2.64,	2.41,	2.25,	2.16,	2.02,	1.93,	1.88,	1.79,
        0.00,	0.00,	0.00,	0.00,	4.97,	4.11,	3.57,	3.30,	2.94,	2.59,	2.39,	2.20,	2.07,	2.00,	1.87,	1.79,	1.75,
        0.00,	0.00,	0.00,	0.00,	4.80,	4.04,	3.46,	3.21,	2.89,	2.63,	2.33,	2.18,	2.02,	1.93,	1.86,	1.75,	1.67,
        0.00,	0.00,	0.00,	0.00,	4.59,	4.01,	3.44,	3.09,	2.88,	2.60,	2.35,	2.15,	2.00,	1.87,	1.80,	1.74,	1.64,
        0.00,	0.00,	0.00,	0.00,	0.00,	3.98,	3.41,	2.99,	2.82,	2.57,	2.36,	2.11,	1.99,	1.85,	1.74,	1.69,	1.63,
        0.00,	0.00,	0.00,	0.00,	0.00,	3.96,	3.36,	2.95,	2.74,	2.56,	2.33,	2.15,	1.94,	1.84,	1.72,	1.63,	1.59,
        0.00,	0.00,	0.00,	0.00,	0.00,	3.88,	3.35,	2.95,	2.63,	2.51,	2.31,	2.14,	1.95,	1.81,	1.72,	1.61,	1.53,
        0.00,	0.00,	0.00,	0.00,	0.00,	3.68,	3.31,	2.91,	2.59,	2.46,	2.31,	2.11,	1.97,	1.79,	1.69,	1.60,	1.51,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	3.33,	2.88,	2.58,	2.35,	2.27,	2.10,	1.95,	1.82,	1.67,	1.60,	1.50,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	3.26,	2.88,	2.58,	2.31,	2.22,	2.10,	1.93,	1.81,	1.67,	1.56,	1.50,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	3.23,	2.84,	2.53,	2.30,	2.14,	2.07,	1.93,	1.80,	1.69,	1.55,	1.48,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	3.04,	2.85,	2.52,	2.30,	2.09,	2.02,	1.93,	1.78,	1.68,	1.58,	1.46,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.84,	2.52,	2.28,	2.07,	1.96,	1.90,	1.78,	1.66,	1.58,	1.46,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.77,	2.49,	2.25,	2.07,	1.92,	1.85,	1.78,	1.65,	1.57,	1.48,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.74,	2.49,	2.24,	2.07,	1.88,	1.81,	1.75,	1.65,	1.55,	1.48,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.56,	2.50,	2.24,	2.04,	1.88,	1.76,	1.72,	1.65,	1.54,	1.46,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.44,	2.21,	2.02,	1.88,	1.74,	1.69,	1.62,	1.54,	1.45,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.43,	2.21,	2.02,	1.88,	1.73,	1.62,	1.60,	1.54,	1.45,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.38,	2.22,	2.02,	1.86,	1.72,	1.61,	1.58,	1.52,	1.44,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.22,	2.21,	2.00,	1.84,	1.72,	1.60,	1.51,	1.49,	1.44,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.16,	1.99,	1.83,	1.72,	1.60,	1.51,	1.48,	1.42,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.16,	2.00,	1.83,	1.69,	1.59,	1.48,	1.43,	1.40,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	2.10,	2.00,	1.82,	1.69,	1.59,	1.48,	1.41,	1.39,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.96,	1.96,	1.81,	1.68,	1.58,	1.48,	1.39,	1.35,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.94,	1.81,	1.68,	1.56,	1.48,	1.38,	1.32,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.94,	1.82,	1.67,	1.55,	1.48,	1.38,	1.32,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.88,	1.81,	1.66,	1.55,	1.46,	1.38,	1.30,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.74,	1.76,	1.66,	1.55,	1.45,	1.38,	1.30,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.76,	1.67,	1.55,	1.44,	1.38,	1.30,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.76,	1.67,	1.53,	1.44,	1.36,	1.30,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.69,	1.64,	1.53,	1.44,	1.35,	1.30,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.55,	1.62,	1.53,	1.44,	1.35,	1.29,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.62,	1.54,	1.42,	1.34,	1.27,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.61,	1.53,	1.42,	1.34,	1.27,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.55,	1.50,	1.42,	1.34,	1.27,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.43,	1.50,	1.43,	1.33,	1.26,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.49,	1.43,	1.33,	1.26,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.47,	1.41,	1.33,	1.26,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.43,	1.39,	1.34,	1.25,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.27,	1.39,	1.34,	1.25,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.39,	1.33,	1.25,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.36,	1.30,	1.25,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.33,	1.30,	1.25,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.16,	1.30,	1.25,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.30,	1.23,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.25,	1.22,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.24,	1.22,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.06,	1.22,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.21,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.17,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.16,
        0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.97    
    };

    distribution = malloc(sizeof(int) * 3);
    if(num_threads * num_procs == 1){
        distribution[0] = num_iterations;
        distribution[1] = 0;
        distribution[2] = num_iterations;
        return distribution; 
    }

    my_row = (num_threads * proc_id) + thread_id;
    my_column = (num_procs * num_threads) / 4;
    my_working_ratio = working_ratios[my_row][my_column];

    block_size = my_working_ratio * num_iterations / 100;
    block_start = 0;
    for(i = 0; i < my_row; i ++){
        block_start += working_ratios[i][my_column] * num_iterations / 100;
    }
    block_end = block_start + block_size;

    if (thread_id == (num_threads - 1) && proc_id == (num_procs - 1)){ 
        //If Last thread from last process:
        block_end = num_iterations;
        block_size = block_end - block_start;
    }

    distribution[0] = block_size;
    distribution[1] = block_start;    
    distribution[2] = block_end;    

    return distribution;
}


/*
 * This method is used by ParallelChudnovskyAlgorithm procs
 * for computing the first value of dep_a
 */
void init_dep_a(mpf_t dep_a, int block_start){
    mpz_t factorial_n, dividend, divisor;
    mpf_t float_dividend, float_divisor;
    mpz_inits(factorial_n, dividend, divisor, NULL);
    mpf_inits(float_dividend, float_divisor, NULL);

    mpz_fac_ui(factorial_n, block_start);
    mpz_fac_ui(divisor, 3 * block_start);
    mpz_fac_ui(dividend, 6 * block_start);

    mpz_pow_ui(factorial_n, factorial_n, 3);
    mpz_mul(divisor, divisor, factorial_n);

    mpf_set_z(float_dividend, dividend);
    mpf_set_z(float_divisor, divisor);

    mpf_div(dep_a, float_dividend, float_divisor);

    mpz_clears(factorial_n, dividend, divisor, NULL);
    mpf_clears(float_dividend, float_divisor, NULL);
}

/*
 * Parallel Pi number calculation using the Chudnovsky algorithm
 * The number of iterations is divided by blocks 
 * so each process calculates a part of pi with multiple threads (or just one thread). 
 * Each process will also divide the iterations in blocks
 * among the threads to calculate its part.  
 * Finally, a collective reduction operation will be performed 
 * using a user defined function in OperationsMPI. 
 */
void Chudnovsky_algorithm_MPI(int num_procs, int proc_id, mpf_t pi, 
                                    int num_iterations, int num_threads){
    int packet_size, position; 
    mpf_t local_proc_pi, e, c;  

    mpf_init_set_ui(local_proc_pi, 0);   
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, thread_block_size, thread_block_start, thread_block_end, factor_a;
        int *distribution;
        mpf_t local_thread_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;

        thread_id = omp_get_thread_num();
        distribution = get_distribution(num_procs, proc_id, num_threads, thread_id, num_iterations);
        thread_block_size = distribution[0];
        thread_block_start = distribution[1];
        thread_block_end = distribution[2];

        mpf_init_set_ui(local_thread_pi, 0);    // private thread pi
        mpf_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, aux, NULL);
        init_dep_a(dep_a, thread_block_start);
        mpf_pow_ui(dep_b, c, thread_block_start);
        mpf_init_set_ui(dep_c, B);
        mpf_mul_ui(dep_c, dep_c, thread_block_start);
        mpf_add_ui(dep_c, dep_c, A);
        factor_a = 12 * thread_block_start;

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_block_start; i < thread_block_end; i++){
                Chudnovsky_iteration(local_thread_pi, i, dep_a, dep_b, dep_c, aux);
                //Update dep_a:
                mpf_set_ui(dep_a_dividend, factor_a + 10);
                mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
                mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
                mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);

                mpf_set_ui(dep_a_divisor, i + 1);
                mpf_pow_ui(dep_a_divisor, dep_a_divisor ,3);
                mpf_div(dep_a, dep_a_dividend, dep_a_divisor);
                factor_a += 12;

                //Update dep_b:
                mpf_mul(dep_b, dep_b, c);

                //Update dep_c:
                mpf_add_ui(dep_c, dep_c, B);
            }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear thread memory
        mpf_clears(local_thread_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux, NULL);   
    }
    
    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)add, 0, &add_op);

    //Set buffers for cumunications and position for pack and unpack information 
    packet_size = 8 + sizeof(mp_exp_t) + ((local_proc_pi -> _mp_prec + 1) * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];

    //Pack local_proc_pi in sendbuffuer
    position = pack(sendbuffer, local_proc_pi);

    //Reduce local_proc_pi
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operations to get Pi
    if (proc_id == 0){
        unpack(recbuffer, pi);
        mpf_sqrt(e, e);
        mpf_mul_ui(e, e, D);
        mpf_div(pi, e, pi); 
    }    

    //Clear process memory
    MPI_Op_free(&add_op);
    mpf_clears(local_proc_pi, e, c, NULL);
}

