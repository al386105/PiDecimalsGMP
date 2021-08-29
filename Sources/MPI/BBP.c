#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "../../Headers/Sequential/BBP.h"
#include "../../Headers/MPI/OperationsMPI.h"


#define QUOTIENT 0.0625

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Last version of Bailey Borwein Plouffe formula implementation                    *
 * It implements a sequential method and another one that can use multiple          *
 * processes and threads in hybrid way.                                             *
 *                                                                                  *
 ************************************************************************************
 * Bailey Borwein Plouffe formula:                                                  *
 *                      1        4          2        1       1                      *
 *    pi = SUMMATORY( ------ [ ------  - ------ - ------ - ------]),  n >=0         *
 *                     16^n    8n + 1    8n + 4   8n + 5   8n + 6                   *
 *                                                                                  *
 * Formula quotients are coded as:                                                  *
 *              4                 2                 1                 1             *
 *   quot_a = ------,  quot_b = ------,  quot_c = ------,  quot_d = ------,         *
 *            8n + 1            8n + 4            8n + 5            8n + 6          *
 *                                                                                  *
 *              1                                                                   *
 *   quot_m = -----                                                                 *
 *             16^n                                                                 *
 *                                                                                  *
 ************************************************************************************
 * Bailey Borwein Plouffe formula dependencies:                                     *
 *                                                                                  *
 *           dep_a(n) = 8n = dep_a(n-1) + 8                                         *
 *                                                                                  *
 *                        1            1                                            *
 *           dep_m(n) = ----- = ---------------                                     *
 *                       16^n   dep_m(n-1) * 16                                     *
 *                                                                                  *
 ************************************************************************************/


/*
 * Parallel Pi number calculation using the BBP algorithm
 * Multiple procs and threads can be used
 * The number of iterations is divided by blocks, 
 * so each process calculates a part of pi using threads. 
 * Each process will cyclically divide the iterations 
 * among the threads to calculate its part.  
 * Finally, a collective reduction operation will be performed
 * using a user defined function in OperationsMPI. 
 */
void BBP_algorithm_MPI(int num_procs, int proc_id, mpf_t pi, 
                            int num_iterations, int num_threads){
    int block_size, block_start, block_end, position, packet_size;
    mpf_t local_proc_pi, jump, quotient;

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = (proc_id == num_procs - 1) ? num_iterations : block_start + block_size;

    mpf_init_set_ui(local_proc_pi, 0);          
    mpf_init_set_d(quotient, QUOTIENT);             // quotient = (1 / 16)   
    mpf_init_set_ui(jump, 1);        
    mpf_pow_ui(jump, quotient, num_threads);        // jump = (1/16)^num_threads
    


    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int thread_id, i;
        mpf_t local_thread_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_thread_pi, 0);                    // private thread pi
        mpf_init(dep_m);      
        mpf_pow_ui(dep_m, quotient, block_start + thread_id);   // dep_m = (1/16)^n      
        mpf_inits(quot_a, quot_b, quot_c, quot_d, aux, NULL);    

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = block_start + thread_id; i < block_end; i+=num_threads){    
                BBP_iteration(local_thread_pi, i, dep_m, quot_a, quot_b, quot_c, quot_d, aux); 
                // Update depencies: 
                mpf_mul(dep_m, dep_m, jump);    
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear memory
        mpf_clears(local_thread_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux, NULL);
    }


    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)add, 0, &add_op);

    //Set buffers for cumunications, and position for pack and unpack information 
    packet_size = 8 + sizeof(mp_exp_t) + ((local_proc_pi -> _mp_prec + 1) * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];


    //Pack local_proc_pi in sendbuffuer
    position = pack(sendbuffer, local_proc_pi);

    //Reduce piLocal
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi
    if (proc_id == 0){
        unpack(recbuffer, pi);
    }


    //Clear memory
    MPI_Op_free(&add_op);
    mpf_clears(local_proc_pi, quotient, jump, NULL);
}



