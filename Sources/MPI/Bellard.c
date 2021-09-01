#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "../../Headers/Sequential/Bellard.h"
#include "../../Headers/MPI/OperationsMPI.h"



/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Bellard formula implementation                                                   *
 * It implements a sequential method and another one that can use multiple          *
 * processes and threads in hybrid way.                                             *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula:                                                                 *
 *                 (-1)^n     32     1      256     64       4       4       1      *
 * 2^6 * pi = SUM( ------ [- ---- - ---- + ----- - ----- - ----- - ----- + -----])  *
 *                 2^10n     4n+1   4n+3   10n+1   10n+3   10n+5   10n+7   10n+9    *
 *                                                                                  *
 * Formula quotients are coded as:                                                  *
 *             32          1           256          64                              *
 *        a = ----,   b = ----,   c = -----,   d = -----,                           *
 *            4n+1        4n+3        10n+1        10n+3                            *
 *                                                                                  *
 *              4            4            1         (-1)^n                          *
 *        e = -----,   f = -----,   g = -----,   m = -----,                         *
 *            10n+5        10n+7        10n+9        2^10n                          *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula dependencies:                                                    *
 *                           1            1                                         *
 *              dep_m(n) = ------ = -----------------                               *
 *                         1024^n   1024^(n-1) * 1024                               *
 *                                                                                  *
 *              dep_a(n) = 4n  = dep_a(n-1) + 4                                     *
 *                                                                                  *
 *              dep_b(n) = 10n = dep_a(n-1) + 10                                    *
 *                                                                                  *
 ************************************************************************************/


/*
 * Parallel Pi number calculation using the Bellard algorithm
 * The number of iterations is divided by blocks, 
 * so each process calculates a part of pi using threads. 
 * Each process will cyclically divide the iterations 
 * among the threads to calculate its part.  
 * Finally, a collective reduction operation will be performed
 * using a user defined function in OperationsMPI. 
 */
void Bellard_algorithm_MPI(int num_procs, int proc_id, mpf_t pi, 
                                int num_iterations, int num_threads){
    int block_size, block_start, block_end, position, packet_size;
    mpf_t local_proc_pi, jump;

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = block_start + block_size;
    if (block_end > num_iterations) block_end = num_iterations;

    mpf_init_set_ui(local_proc_pi, 0);
    mpf_init_set_ui(jump, 1); 
    mpf_div_ui(jump, jump, 1024);
    mpf_pow_ui(jump, jump, num_threads);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, dep_a, dep_b, jump_dep_a, jump_dep_b;
        mpf_t local_thread_pi, dep_m, a, b, c, d, e, f, g, aux;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_thread_pi, 0);       // private thread pi
        dep_a = (block_start + thread_id) * 4;
        dep_b = (block_start + thread_id) * 10;
        jump_dep_a = 4 * num_threads;
        jump_dep_b = 10 * num_threads;
        mpf_init_set_ui(dep_m, 1);
        mpf_div_ui(dep_m, dep_m, 1024);
        mpf_pow_ui(dep_m, dep_m, block_start + thread_id);  // dep_m = ((-1)^n)/1024)
        if((block_start + thread_id) % 2 != 0) mpf_neg(dep_m, dep_m);                   
        mpf_inits(a, b, c, d, e, f, g, aux, NULL);

        //First Phase -> Working on a local variable
        if(num_threads % 2 != 0){
            #pragma omp parallel for 
                for(i = block_start + thread_id; i < block_end; i+=num_threads){
                    Bellard_iteration(local_thread_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
                    // Update dependencies for next iteration:
                    mpf_mul(dep_m, dep_m, jump); 
                    mpf_neg(dep_m, dep_m); 
                    dep_a += jump_dep_a;
                    dep_b += jump_dep_b;     
                }
        } else {
            #pragma omp parallel for
                for(i = block_start + thread_id; i < block_end; i+=num_threads){
                    Bellard_iteration(local_thread_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
                    // Update dependencies for next iteration:
                    mpf_mul(dep_m, dep_m, jump);    
                    dep_a += jump_dep_a;
                    dep_b += jump_dep_b;    
                }
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear memory
        mpf_clears(local_thread_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);   
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

    //Reduce piLocal
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operation
    if (proc_id == 0){
        unpack(recbuffer, pi);
        mpf_div_ui(pi, pi, 64);
    }

    //Clear memory
    MPI_Op_free(&add_op);
    mpf_clears(local_proc_pi, jump, NULL);       
}

