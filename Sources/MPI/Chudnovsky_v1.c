#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "../../Headers/Sequential/Chudnovsky_v1.h"
#include "../../Headers/MPI/OperationsMPI.h"


#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Chudnovsky formula implementation                                                *
 * This version computes all the factorials needed before performing the iterations *
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
 *      dividend = (6n)! (545140134n + 13591409)                                    *
 *      divisor  = (n!)^3 (3n)! (-640320)^3n                                        *
 *      e        = 426880 sqrt(10005)                                               *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula dependencies:                                                 *
 *              dep_a(n) = (6n)!                                                    *
 *              dep_b(n) = (n!)^3                                                   *
 *              dep_c(n) = (3n)!                                                    *
 *              dep_d(n) = (-640320)^(3n) = (-640320)^(3 (n-1)) * (-640320)^3       *
 *              dep_e(n) = (545140134n + 13591409) = dep_c(n - 1) + 545140134       *
 *                                                                                  *
 ************************************************************************************/


/*
 * Parallel Pi number calculation using the Chudnovsky algorithm
 * The number of iterations is divided by blocks 
 * so each process calculates a part of pi with multiple threads (or just one thread). 
 * Each process will also divide the iterations in blocks
 * among the threads to calculate its part.  
 * Finally, a collective reduction operation will be performed 
 * using a user defined function in OperationsMPI. 
 */
void Chudnovsky_algorithm_v1_MPI(int num_procs, int proc_id, mpf_t pi, 
                                    int num_iterations, int num_threads){
    int block_size, block_start, block_end, num_factorials, packet_size, position; 
    mpf_t local_proc_pi, e, c;

    //All procs computes factorials
    num_factorials = num_iterations * 6;
    mpf_t factorials[num_factorials + 1];
    get_factorials(factorials, num_factorials);

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = (proc_id == num_procs - 1) ? num_iterations : block_start + block_size;

    mpf_init_set_ui(local_proc_pi, 0);   
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, thread_block_size, thread_start, thread_end;
        mpf_t local_thread_pi, dep_a, dep_b, dep_c, dep_d, dep_e, dividend, divisor;

        thread_id = omp_get_thread_num();
        thread_block_size = (block_size + num_threads - 1) / num_threads;
        thread_start = (thread_id * thread_block_size) + (block_size * proc_id);
        thread_end = (thread_id == num_threads - 1) ? block_end : thread_start + thread_block_size;
        
        mpf_init_set_ui(local_thread_pi, 0);    // private thread pi
        mpf_inits(dividend, divisor, NULL);
        mpf_init_set(dep_a, factorials[thread_start * 6]);
        mpf_init_set(dep_b, factorials[thread_start]);
        mpf_pow_ui(dep_b, dep_b, 3);
        mpf_init_set(dep_c, factorials[thread_start * 3]);
        mpf_init_set_ui(dep_d, C);
        mpf_neg(dep_d, dep_d);
        mpf_pow_ui(dep_d, dep_d, thread_start * 3);
        mpf_init_set_ui(dep_e, B);
        mpf_mul_ui(dep_e, dep_e, thread_start);
        mpf_add_ui(dep_e, dep_e, A);

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_start; i < thread_end; i++){
                Chudnovsky_iteration_v1(local_thread_pi, i, dep_a, dep_b, dep_c, dep_d, dep_e, dividend, divisor);
                //Update dependencies:
                mpf_set(dep_a, factorials[6 * (i + 1)]);
                mpf_pow_ui(dep_b, factorials[i + 1], 3); 
                mpf_set(dep_c, factorials[3 * (i + 1)]);
                mpf_mul(dep_d, dep_d, c);
                mpf_add_ui(dep_e, dep_e, B);
            }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear thread memory
        mpf_clears(local_thread_pi, dep_a, dep_b, dep_c, dep_d, dividend, divisor, NULL);   
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
    clear_factorials(factorials, num_factorials);
    mpf_clears(local_proc_pi, e, c, NULL);
}

