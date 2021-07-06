#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

#define QUOTIENT 0.0625

/*Bailey Borwein Plouffe formula
*************************************************************************************
*                      1        4          2        1       1                       * 
*    pi = SUMMATORY( ------ [ ------  - ------ - ------ - ------]),  n >=0          *
*                     16^n    8n + 1    8n + 4   8n + 5   8n + 6                    *
*************************************************************************************/

/*
 * An iteration of Bailey Borwein Plouffe formula
 */
void BBPIteration(mpf_t pi, int n, mpf_t m, mpf_t a, mpf_t b, mpf_t c, mpf_t d, mpf_t aux ){
    mpf_set_ui(a, 4.0);         // a = (  4 / (8n + 1))
    mpf_set_ui(b, 2.0);         // b = ( -2 / (8n + 4))
    mpf_set_ui(c, 1.0);         // c = ( -1 / (8n + 5))
    mpf_set_ui(d, 1.0);         // d = ( -1 / (8n + 6))
    mpf_set_ui(aux, 0);         // aux = (a + b + c + d)  

    int i = n << 3;             // i = n * 8 
    mpf_div_ui(a, a, i | 1);    // a = (4 / (8n + 1)), i + 1 => i | 1
    mpf_div_ui(b, b, i | 4);    // b = (2 / (8n + 4)), i + 4 => i | 4
    mpf_div_ui(c, c, i | 5);    // c = (1 / (8n + 5)), i + 5 => i | 5
    mpf_div_ui(d, d, i | 6);    // d = (1 / (8n + 6)), i + 6 => i | 6

    // aux = (a - b - c - d)   
    mpf_sub(aux, a, b);
    mpf_sub(aux, aux, c);
    mpf_sub(aux, aux, d);

    // aux = m * aux = m * (a - b - c - d)
    mpf_mul(aux, aux, m);   
    
    mpf_add(pi, pi, aux);    
}

/*
 * Sequential Pi number calculation using the BBP algorithm
 */
void SequentialBBPAlgorithm(mpf_t pi, int num_iterations){   
    int i;
    mpf_t m, quotient, a, b, c, d, aux;           
    mpf_init_set_ui(m, 1);              // m = (1/16)^n
    mpf_init_set_d(quotient, QUOTIENT); // quotient = (1/16)   
    mpf_inits(a, b, c, d, aux, NULL);

    for(i = 0; i < num_iterations; i++){ 
        BBPIteration(pi, i, m, a, b, c, d, aux);   
        // Update m for next iteration: m^n = m^num_threads * m^(n-num_threads) 
        mpf_mul(m, m, quotient);
    }
    mpf_clears(m, quotient, a, b, c, d, aux, NULL);
}

/*
 * Parallel Pi number calculation using the BBP algorithm
 * The number of iterations is divided cyclically, 
 * so each thread calculates a part of Pi.  
 */
void ParallelBBPAlgorithm(mpf_t pi, int num_iterations, int num_threads){
    mpf_t jump, quotient; 
    mpf_init_set_d(quotient, QUOTIENT);         // quotient = (1 / 16)   
    mpf_init_set_ui(jump, 1);        
    mpf_pow_ui(jump, quotient, num_threads);    // jump = (1/16)^num_threads

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i;
        mpf_t local_pi, m, a, b, c, d, aux;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_pi, 0);       // private thread pi
        mpf_init(m);
        mpf_pow_ui(m, quotient, thread_id); // m = (1/16)^n                  
        mpf_inits(a, b, c, d, aux, NULL);

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_id; i < num_iterations; i+=num_threads){
                BBPIteration(local_pi, i, m, a, b, c, d, aux);
                // Update m for next iteration: m^n = m^num_threads * m^(n-num_threads) 
                mpf_mul(m, m, jump);    
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(pi, pi, local_pi);

        //Clear memory
        mpf_clears(local_pi, m, a, b, c, d, aux, NULL);   
    }
        
    //Clear memory
    mpf_clears(jump, quotient, NULL);
}

