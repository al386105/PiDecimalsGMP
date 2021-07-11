#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005


/************************************************************************************
 * Chudnovsky formula implementation                                                *
 * This version does not computes the factorials                                    *
 * Only implements the single thread version                                        *
 ************************************************************************************
 * Chudnovsky formula:                                                              *
 *     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)             * 
 *    --------------------  = SUMMATORY( ----------------------------- ),  n >=0    *
 *            pi                            (n!)^3 (3n)! (-640320)^3n               *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula dependencies:                                                 *
 *                     (6n)!         (12n + 10)(12n + 6)(12n + 2)                   *
 *      dep_a(n) = --------------- = ---------------------------- * dep_a(n-1)      * 
 *                 ((n!)^3 (3n)!)              (n + 1)^3                            *
 *                                                                                  *
 *      dep_b(n) = (-640320)^3n = (-640320)^3(n-1) * (-640320)^3)                   *
 *                                                                                  *
 ************************************************************************************/


/*
 * An iteration of Chudnovsky formula
 */
void ChudnovskyIterationV1(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, 
                            mpf_t dividend, mpf_t divisor){
    mpf_set_ui(dividend, n);
    mpf_mul_ui(dividend, dividend, B);
    mpf_add_ui(dividend, dividend, A);
    mpf_mul(dividend, dividend, dep_a);

    mpf_set(divisor, dep_b);
    
    mpf_div(dividend, dividend, divisor);
    mpf_add(pi, pi, dividend);
}

/*
 * Sequential Pi number calculation using the Chudnovsky algorithm
 * Single thread implementation
 */
void SequentialChudnovskyAlgorithmV1(mpf_t pi, int num_iterations){
    int i, factor_a;
    mpf_t dep_a, dep_a_dividend, dep_a_divisor, dep_b, aux, c, dividend, divisor;

    mpf_inits(dep_a_dividend, dep_a_divisor, dividend, divisor, NULL);
    mpf_init_set_ui(dep_a, 1);
    mpf_init_set_ui(dep_b, 1);
    mpf_init_set_ui(aux, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    for(i = 0; i < num_iterations; i ++){
        ChudnovskyIterationV1(pi, i, dep_a, dep_b, dividend, divisor);
        //Update dependencies
        factor_a = 12 * i;
        mpf_set_ui(dep_a_dividend, factor_a + 10);
        mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
        mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
        mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);

        mpf_set_ui(dep_a_divisor, i + 1);
        mpf_pow_ui(dep_a_divisor, dep_a_divisor ,3);
        mpf_div(dep_a, dep_a_dividend, dep_a_divisor);
 
        mpf_mul(dep_b, dep_b, c);
    }

    mpf_sqrt(aux, aux);
    mpf_mul_ui(aux, aux, D);
    mpf_div(pi, aux, pi);    
    
    //Clear memory
    mpf_clears(dep_a, dep_b, c, aux, dividend, divisor, NULL);
}