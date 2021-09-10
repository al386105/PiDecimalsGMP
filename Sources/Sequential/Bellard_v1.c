#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>


/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Bellard formula implementation                                                   *
 * It implements a single-threaded method and another that can use multiple threads *
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
 * An iteration of Bellard formula
 */
void Bellard_iteration(mpf_t pi, int n, mpf_t m, mpf_t a, mpf_t b, mpf_t c, mpf_t d, 
                    mpf_t e, mpf_t f, mpf_t g, mpf_t aux, int dep_a, int dep_b){
    mpf_set_ui(a, 32);              // a = ( 32 / ( 4n + 1))
    mpf_set_ui(b, 1);               // b = (  1 / ( 4n + 3))
    mpf_set_ui(c, 256);             // c = (256 / (10n + 1))
    mpf_set_ui(d, 64);              // d = ( 64 / (10n + 3))
    mpf_set_ui(e, 4);               // e = (  4 / (10n + 5))
    mpf_set_ui(f, 4);               // f = (  4 / (10n + 7))
    mpf_set_ui(g, 1);               // g = (  1 / (10n + 9))
    mpf_set_ui(aux, 0);             // aux = (- a - b + c - d - e - f + g)  

    mpf_div_ui(a, a, dep_a + 1);    // a = ( 32 / ( 4n + 1))
    mpf_div_ui(b, b, dep_a + 3);    // b = (  1 / ( 4n + 3))

    mpf_div_ui(c, c, dep_b + 1);    // c = (256 / (10n + 1))
    mpf_div_ui(d, d, dep_b + 3);    // d = ( 64 / (10n + 3))
    mpf_div_ui(e, e, dep_b + 5);    // e = (  4 / (10n + 5))
    mpf_div_ui(f, f, dep_b + 7);    // f = (  4 / (10n + 7))
    mpf_div_ui(g, g, dep_b + 9);    // g = (  1 / (10n + 9))

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
 * Single thread implementation
 */
void Bellard_algorithm_v1(mpf_t pi, int num_iterations){   
    int i, dep_a, dep_b;
    mpf_t dep_m, jump, a, b, c, d, e, f, g, aux;    

    dep_a = 0, dep_b = 0;       
    mpf_init_set_d(jump, 1);            // jump = 1/1024  
    mpf_div_ui(jump, jump, 1024); 
    mpf_init_set_ui(dep_m, 1);          // dep_m = ((-1)^n)/1024)
    mpf_inits(a, b, c, d, e, f, g, aux, NULL);

    for(i = 0; i < num_iterations; i++){ 
        Bellard_iteration(pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);   
        // Update dependencies for next iteration: 
        mpf_mul(dep_m, dep_m, jump);
        mpf_neg(dep_m, dep_m);
        dep_a += 4;
        dep_b += 10;
    }

    mpf_div_ui(pi, pi, 64);
    
    mpf_clears(dep_m, jump, a, b, c, d, e, f, g, aux, NULL);
}

