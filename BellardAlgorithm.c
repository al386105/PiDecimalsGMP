#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>


/************************************************************************************
 * Bellard formula implementation                                                   *
 *                                                                                  *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula:                                                                 *
 *                 (-1)^n     32     1      256     64       4       4       1      *
 * 2^6 * pi = SUM( ------ [- ---- - ---- + ----- - ----- - ----- - ----- + -----])  *
 *                 2^10n     4n+1   4n+3   10n+1   10n+3   10n+5   10n+7   10n+9    *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula dependencies:                                                    *
 *                           1            1                                         *
 *                  m(n) = ------ = -----------------                               *
 *                         1024^n   1024^(n-1) * 1024                               *
 *                                                                                  *                                          *
 ************************************************************************************/


/*
 * An iteration of Bellard formula
 */
void BellardIteration(mpf_t pi, int n, mpf_t m, mpf_t a, mpf_t b, mpf_t c, mpf_t d, 
                        mpf_t e, mpf_t f, mpf_t g, mpf_t aux ){
    mpf_set_ui(a, 32);          // a = ( 32 / ( 4n + 1))
    mpf_set_ui(b, 1);           // b = (  1 / ( 4n + 3))
    mpf_set_ui(c, 256);         // c = (256 / (10n + 1))
    mpf_set_ui(d, 64);          // d = ( 64 / (10n + 3))
    mpf_set_ui(e, 4);           // e = (  4 / (10n + 5))
    mpf_set_ui(f, 4);           // f = (  4 / (10n + 7))
    mpf_set_ui(g, 1);           // g = (  1 / (10n + 9))
    mpf_set_ui(aux, 0);         // aux = (- a - b + c - d - e - f + g)  

    int i = n * 4;              // i = n * 4 
    mpf_div_ui(a, a, i + 1);    // a = ( 32 / ( 4n + 1))
    mpf_div_ui(b, b, i + 3);    // b = (  1 / ( 4n + 3))

    i = n * 10;                 // i = n * 10
    mpf_div_ui(c, c, i + 1);    // c = (256 / (10n + 1))
    mpf_div_ui(d, d, i + 3);    // d = ( 64 / (10n + 3))
    mpf_div_ui(e, e, i + 5);    // e = (  4 / (10n + 5))
    mpf_div_ui(f, f, i + 7);    // f = (  4 / (10n + 7))
    mpf_div_ui(g, g, i + 9);    // g = (  1 / (10n + 9))

    // aux = (- a - b + c - d - e - f + g)   
    mpf_neg(a, a);
    mpf_sub(aux, a, b);
    mpf_sub(c, c, d);
    mpf_sub(c, c, e);
    mpf_sub(c, c, f);
    mpf_add(c, c, g);
    mpf_add(aux, aux, c);

    // aux = m * aux
    mpf_mul(aux, aux, m);   

    mpf_add(pi, pi, aux); 
}

/*
 * Sequential Pi number calculation using the Bellard algorithm
 */
void SequentialBellardAlgorithm(mpf_t pi, int num_iterations){   
    int i;
    mpf_t m, jump, a, b, c, d, e, f, g, aux;           
    mpf_init_set_d(jump, 1);        // jump = 1/1024  
    mpf_div_ui(jump, jump, 1024); 
    mpf_init_set_ui(m, 1);          // m = ((-1)^n)/1024)
    mpf_inits(a, b, c, d, e, f, g, aux, NULL);
    for(i = 0; i < num_iterations; i++){ 
        BellardIteration(pi, i, m, a, b, c, d, e, f, g, aux);   
        // Update m for next iteration: 
        mpf_mul(m, m, jump);
        mpf_neg(m, m);
    }

    mpf_div_ui(pi, pi, 64);
    
    mpf_clears(m, jump, a, b, c, d, e, f, g, aux, NULL);
}

/*
 * Parallel Pi number calculation using the Bellard algorithm
 * The number of iterations is divided cyclically, 
 * so each thread calculates a part of Pi.  
 */
void ParallelBellardAlgorithm(mpf_t pi, int num_iterations, int num_threads){
    mpf_t jump; 
    mpf_init_set_ui(jump, 1); 
    mpf_div_ui(jump, jump, 1024);
    mpf_pow_ui(jump, jump, num_threads);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i;
        mpf_t local_pi, m, a, b, c, d, e, f, g, aux;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_pi, 0);       // private thread pi
        mpf_init_set_ui(m, 1);
        mpf_div_ui(m, m, 1024);
        mpf_pow_ui(m, m, thread_id);        // m = ((-1)^n)/1024)
        if(thread_id % 2 != 0) mpf_neg(m, m);                   
        mpf_inits(a, b, c, d, e, f, g, aux, NULL);

        //First Phase -> Working on a local variable
        if(num_threads % 2 != 0){
            #pragma omp parallel for 
                for(i = thread_id; i < num_iterations; i+=num_threads){
                    BellardIteration(local_pi, i, m, a, b, c, d, e, f, g, aux);
                    // Update m for next iteration:
                    mpf_mul(m, m, jump); 
                    mpf_neg(m, m);   
                }
        } else {
            #pragma omp parallel for
                for(i = thread_id; i < num_iterations; i+=num_threads){
                    BellardIteration(local_pi, i, m, a, b, c, d, e, f, g, aux);
                    // Update m for next iteration:
                    mpf_mul(m, m, jump);    
                }
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(pi, pi, local_pi);

        //Clear memory
        mpf_clears(local_pi, m, a, b, c, d, e, f, g, aux, NULL);   
    }

    mpf_div_ui(pi, pi, 64);
        
    //Clear memory
    mpf_clear(jump);
}

