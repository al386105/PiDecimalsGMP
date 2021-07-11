#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>


/************************************************************************************
 * Second version of Bailey Borwein Plouffe formula implementation                  *
 * Does not free the memory                                                         *
 *                                                                                  *
 ************************************************************************************
 * Bailey Borwein Plouffe formula:                                                  *
 *                      1        4          2        1       1                      * 
 *    pi = SUMMATORY( ------ [ ------  - ------ - ------ - ------]),  n >=0         *
 *                     16^n    8n + 1    8n + 4   8n + 5   8n + 6                   *
 *                                                                                  *
 ************************************************************************************
 * Bailey Borwein Plouffe formula dependencies:                                     *
 *                           1          1                                           *
 *                  m(n) = ----- = ------------                                     *
 *                          16^n   16^(n-1) * 16                                    *
 *                                                                                  *
 ************************************************************************************/


/*
 * An iteration of Bailey Borwein Plouffe formula
 */
void BBPIterationV2(mpf_t pi, int n, mpf_t jump, mpf_t m){
    mpf_t a, b, c, d, aux;
    mpf_init_set_ui(a, 4.0);        // (  4 / 8n + 1))
    mpf_init_set_ui(b, 2.0);        // ( -2 / 8n + 4))
    mpf_init_set_ui(c, 1.0);        // ( -1 / 8n + 5))
    mpf_init_set_ui(d, 1.0);        // ( -1 / 8n + 6))
    mpf_init_set_ui(aux, 0);        // (a + b + c + d)  

    int i = n << 3;                 // i = n * 8 
    mpf_div_ui(a, a, i | 1);      // a = 4 / (8n + 1), i + 1 => i | 1
    mpf_div_ui(b, b, i | 4);      // b = 2 / (8n + 4), i + 4 => i | 4
    mpf_div_ui(c, c, i | 5);      // c = 1 / (8n + 5), i + 5 => i | 5
    mpf_div_ui(d, d, i | 6);      // d = 1 / (8n + 6), i + 6 => i | 6

    // aux = (a - b - c - d)   
    mpf_sub(aux, a, b);
    mpf_sub(aux, aux, c);
    mpf_sub(aux, aux, d);

    // aux = m * aux
    mpf_mul(aux, aux, m);   
    
    mpf_add(pi, pi, aux);    

    // Update m for next iteration: m^n = m^num_threads * m^(n-num_threads) 
    mpf_mul(m, m, jump);  
}

/*
 * Sequential Pi number calculation using the BBP algorithm
 * Single thread implementation
 */
void SequentialBBPAlgorithmV2(mpf_t pi, int num_iterations){    
    double q = 1.0 / 16.0;
    mpf_t m, quotient;           
    mpf_init_set_ui(m, 1);          // m = (1/16)^n
    mpf_init_set_d(quotient, q);    // quotient = (1/16)   

    int i;
    for(i = 0; i < num_iterations; i++){
        BBPIterationV2(pi, i, quotient, m);   
    }

}

/*
 * Parallel Pi number calculation using the BBP algorithm
 * Multiple threads can be used
 * The number of iterations is divided cyclically, 
 * so each thread calculates a part of Pi.  
 */
void ParallelBBPAlgorithmV2(mpf_t pi, int num_iterations, int num_threads){
    int thread_id, i;
    double q = 1.0 / 16.0;
    mpf_t jump, quotient; 
    mpf_init_set_d(quotient, q);            // quotient = (1 / 16)   
    mpf_init_set_ui(jump, 1);        
    mpf_pow_ui(jump, quotient, num_threads); // jump = (1/16)^num_threads

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel private(thread_id, i)
    {
        thread_id = omp_get_thread_num();
        mpf_t local_pi, m;
        mpf_init_set_ui(local_pi, 0);    // private thread pi
        mpf_init_set_ui(m, 0);
        mpf_pow_ui(m, quotient, thread_id);  // m = (1/16)^n          
        
        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_id; i < num_iterations; i+=num_threads){
                BBPIterationV2(local_pi, i, jump, m);    
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(pi, pi, local_pi);

        //Clear memory
        mpf_clear(local_pi);   
        mpf_clear(m);     
    }
        
    //Clear memory
    mpf_clear(quotient);
    mpf_clear(jump);
}

