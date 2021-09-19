#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "../../Headers/Sequential/Bellard_v1.h"


/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * First version of Bellard formula                                                 *
 * It allows to compute pi using multiple threads                                   *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula:                                                                 *
 *                 (-1)^n     32     1      256     64       4       4       1      *
 * 2^6 * pi = SUM( ------ [- ---- - ---- + ----- - ----- - ----- - ----- + -----])  *
 *                 1024^n    4n+1   4n+3   10n+1   10n+3   10n+5   10n+7   10n+9    *
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
 * Multiple threads can be used
 * The number of iterations is divided cyclically, 
 * so each thread calculates a part of Pi.  
 */
void Bellard_algorithm_v1_OMP(mpf_t pi, int num_iterations, int num_threads){
    mpf_t jump; 

    mpf_init_set_ui(jump, 1); 
    mpf_div_ui(jump, jump, 1024);
    mpf_pow_ui(jump, jump, num_threads);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, dep_a, dep_b, jump_dep_a, jump_dep_b;
        mpf_t local_pi, dep_m, a, b, c, d, e, f, g, aux;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_pi, 0);       // private thread pi
        dep_a = thread_id * 4;
        dep_b = thread_id * 10;
        jump_dep_a = 4 * num_threads;
        jump_dep_b = 10 * num_threads;
        mpf_init_set_ui(dep_m, 1);
        mpf_div_ui(dep_m, dep_m, 1024);
        mpf_pow_ui(dep_m, dep_m, thread_id);        // dep_m = ((-1)^n)/1024)
        if(thread_id % 2 != 0) mpf_neg(dep_m, dep_m);                   
        mpf_inits(a, b, c, d, e, f, g, aux, NULL);

        //First Phase -> Working on a local variable
        if(num_threads % 2 != 0){
            #pragma omp parallel for 
                for(i = thread_id; i < num_iterations; i+=num_threads){
                    Bellard_iteration(local_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
                    // Update dependencies for next iteration:
                    mpf_mul(dep_m, dep_m, jump); 
                    mpf_neg(dep_m, dep_m); 
                    dep_a += jump_dep_a;
                    dep_b += jump_dep_b;  
                }
        } else {
            #pragma omp parallel for
                for(i = thread_id; i < num_iterations; i+=num_threads){
                    Bellard_iteration(local_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
                    // Update dependencies for next iteration:
                    mpf_mul(dep_m, dep_m, jump);    
                    dep_a += jump_dep_a;
                    dep_b += jump_dep_b;  
                }
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(pi, pi, local_pi);

        //Clear thread memory
        mpf_clears(local_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);   
    }

    mpf_div_ui(pi, pi, 64);
        
    //Clear memory
    mpf_clear(jump);
}

