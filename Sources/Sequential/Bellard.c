#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "../../Headers/Sequential/Bellard_v1.h"


/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Last version of Bellard formula                                                  *
 * It computes pi with a single thread                                              *
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
 * Sequential Pi number calculation using the Bellard algorithm
 * Single thread implementation
 */
void Bellard_algorithm(mpf_t pi, int num_iterations){   
    int i, dep_a, dep_b, next_i;
    mpf_t dep_m, a, b, c, d, e, f, g, aux, ONE;    

    dep_a = 0, dep_b = 0;       
    mpf_init_set_ui(dep_m, 1);          
    mpf_init_set_ui(ONE, 1);
    mpf_inits(a, b, c, d, e, f, g, aux, NULL);

    for(i = 0; i < num_iterations; i++){ 
        Bellard_iteration(pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);   
        // Update dependencies for next iteration: 
        next_i = i + 1;
        mpf_mul_2exp(dep_m, ONE, 10 * next_i);
        mpf_div(dep_m, ONE, dep_m);
        if (next_i % 2 != 0) mpf_neg(dep_m, dep_m); 
        dep_a += 4;
        dep_b += 10;
    }

    mpf_div_ui(pi, pi, 64);
    
    mpf_clears(dep_m, a, b, c, d, e, f, g, aux, ONE, NULL);
}

