#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005


/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Last version of Chudnovsky formula                                               *
 * This version does not computes all the factorials                                *
 * It computes pi with a single thread                                              *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula:                                                              *
 *     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)             *
 *    --------------------  = SUMMATORY( ----------------------------- ),  n >=0    *
 *            pi                            (n!)^3 (3n)! (-640320)^3n               *
 *                                                                                  *
 * Some operands of the formula are coded as:                                       *
 *      dep_a_dividend = (6n)!                                                      *
 *      dep_a_divisor  = (n!)^3 (3n)!                                               *
 *      e              = 426880 sqrt(10005)                                         *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula dependencies:                                                 *
 *                     (6n)!         (12n + 10)(12n + 6)(12n + 2)                   *
 *      dep_a(n) = --------------- = ---------------------------- * dep_a(n-1)      *
 *                 ((n!)^3 (3n)!)              (n + 1)^3                            *
 *                                                                                  *
 *      dep_b(n) = (-640320)^3n = (-640320)^3(n-1) * (-640320)^3)                   *
 *                                                                                  *
 *      dep_c(n) = (545140134n + 13591409) = dep_c(n - 1) + 545140134               *
 *                                                                                  *
 ************************************************************************************/


/*
 * An iteration of Chudnovsky formula
 */
void Chudnovsky_iteration(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, 
                            mpf_t dep_c, mpf_t aux){
    mpf_mul(aux, dep_a, dep_c);
    mpf_div(aux, aux, dep_b);
    
    mpf_add(pi, pi, aux);
}

/*
 * Sequential Pi number calculation using the Chudnovsky algorithm
 * Single thread implementation
 */
void Chudnovsky_algorithm(mpf_t pi, int num_iterations){
    int i, factor_a;
    mpf_t dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, e, c, aux;

    mpf_inits(dep_a_dividend, dep_a_divisor, aux, NULL);
    mpf_init_set_ui(dep_a, 1);
    mpf_init_set_ui(dep_b, 1);
    mpf_init_set_ui(dep_c, A);
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    for(i = 0; i < num_iterations; i ++){
        Chudnovsky_iteration(pi, i, dep_a, dep_b, dep_c, aux);
        //Update dep_a:
        factor_a = 12 * i;
        mpf_set_ui(dep_a_dividend, factor_a + 10);
        mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
        mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
        mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);

        mpf_set_ui(dep_a_divisor, i + 1);
        mpf_pow_ui(dep_a_divisor, dep_a_divisor ,3);
        mpf_div(dep_a, dep_a_dividend, dep_a_divisor);

        //Update dep_b:
        mpf_mul(dep_b, dep_b, c);

        //Update dep_c:
        mpf_add_ui(dep_c, dep_c, B);
    }

    mpf_sqrt(e, e);
    mpf_mul_ui(e, e, D);
    mpf_div(pi, e, pi);    
    
    //Clear memory
    mpf_clears(dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, e, c, aux, NULL);
}

