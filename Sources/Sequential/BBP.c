#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#define QUOTIENT 0.0625

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Last version of Bailey Borwein Plouffe formula                                   *
 * It computes pi with a single thread                                              *
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
 *                        1            1                                            *
 *           dep_m(n) = ----- = ---------------                                     *
 *                       16^n   dep_m(n-1) * 16                                     *
 *                                                                                  *
 ************************************************************************************/

/*
 * An iteration of Bailey Borwein Plouffe formula
 */
void BBP_iteration(mpf_t pi, int n, mpf_t dep_m, 
                mpf_t quot_a, mpf_t quot_b, mpf_t quot_c, mpf_t quot_d, mpf_t aux){
    mpf_set_ui(quot_a, 4);              // quot_a = ( 4 / (8n + 1))
    mpf_set_ui(quot_b, 2);              // quot_b = (-2 / (8n + 4))
    mpf_set_ui(quot_c, 1);              // quot_c = (-1 / (8n + 5))
    mpf_set_ui(quot_d, 1);              // quot_d = (-1 / (8n + 6))
    mpf_set_ui(aux, 0);                 // aux = a + b + c + d  

    int i = n << 3;                     // i = 8n
    mpf_div_ui(quot_a, quot_a, i | 1);  // 4 / (8n + 1)
    mpf_div_ui(quot_b, quot_b, i | 4);  // 2 / (8n + 4)
    mpf_div_ui(quot_c, quot_c, i | 5);  // 1 / (8n + 5)
    mpf_div_ui(quot_d, quot_d, i | 6);  // 1 / (8n + 6)

    // aux = (a - b - c - d)   
    mpf_sub(aux, quot_a, quot_b);
    mpf_sub(aux, aux, quot_c);
    mpf_sub(aux, aux, quot_d);

    // aux = m * aux 
    mpf_mul(aux, aux, dep_m);   
    
    mpf_add(pi, pi, aux);  
}

/*
 * Sequential Pi number calculation using the BBP algorithm
 * Single thread implementation
 */
void BBP_algorithm(mpf_t pi, int num_iterations){   
    int i;
    mpf_t dep_m, quotient, quot_a, quot_b, quot_c, quot_d, aux;

    mpf_inits(quot_a, quot_b, quot_c, quot_d, aux, NULL);
    mpf_init_set_ui(dep_m, 1);          // m = (1/16)^n
    mpf_init_set_d(quotient, QUOTIENT); // quotient = (1/16)   

    for(i = 0; i < num_iterations; i++){ 
        BBP_iteration(pi, i, dep_m, quot_a, quot_b, quot_c, quot_d, aux);   
        // Update dependencies:  
        mpf_mul(dep_m, dep_m, quotient);
    }

    mpf_clears(dep_m, quotient, quot_a, quot_b, quot_c, quot_d, aux, NULL);
}

