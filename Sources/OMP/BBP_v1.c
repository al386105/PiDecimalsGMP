#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "../../Headers/Sequential/BBP_v1.h"


#define QUOTIENT 0.0625

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * First version of Bailey Borwein Plouffe formula implementation                   *
 * It implements a single-threaded method and another that can use multiple threads *
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
 *   quot_m = ------                                                                *
 *             16^n                                                                 *
 *                                                                                  *
 ************************************************************************************


/*
 * Parallel Pi number calculation using the BBP algorithm
 * The number of iterations is divided cyclically, 
 * so each thread calculates a part of Pi.  
 */
void BBP_algorithm_v1_OMP(mpf_t pi, int num_iterations, int num_threads){
    int thread_id, i;
    mpf_t quotient; 

    mpf_init_set_d(quotient, QUOTIENT); // quotient = (1 / 16)   

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel private(thread_id, i)
    {
        mpf_t local_pi;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_pi, 0);   // private thread pi
        
        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_id; i < num_iterations; i+=num_threads){
                BBP_iteration_v1(local_pi, i, quotient);    
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(pi, pi, local_pi);

        //Clear thread memory
        mpf_clear(local_pi);   
    }
        
    //Clear memory
    mpf_clear(quotient);
}
