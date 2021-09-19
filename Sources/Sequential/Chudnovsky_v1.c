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
 * First version of Chudnovsky formula                                              *
 * This version computes all the factorials needed before performing the iterations *
 * It computes pi with a single thread                                              *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula:                                                              *
 *     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)             *
 *    --------------------  = SUMMATORY( ----------------------------- ),  n >=0    *
 *            pi                            (n!)^3 (3n)! (-640320)^3n               *
 *                                                                                  *
 * Some operands of the formula are coded as:                                       *
 *      dividend = (6n)! (545140134n + 13591409)                                    *
 *      divisor  = (n!)^3 (3n)! (-640320)^3n                                        *
 *      e        = 426880 sqrt(10005)                                               *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula dependencies:                                                 *
 *              dep_a(n) = (6n)!                                                    *
 *              dep_b(n) = (n!)^3                                                   *
 *              dep_c(n) = (3n)!                                                    *
 *              dep_d(n) = (-640320)^(3n) = (-640320)^(3 (n-1)) * (-640320)^3       *
 *              dep_e(n) = (545140134n + 13591409) = dep_c(n - 1) + 545140134       *
 *                                                                                  *
 ************************************************************************************/

/*
 * This method calculates the factorials from 0 to num_factorials (included) 
 * and stores them in their corresponding vector position (factorials[n] = n!): 
 * factorials[0] = 1, factorials[1] = 1, factorials[2] = 2, factorials[3] = 6, etc.
 * The computation is performed with a single thread. 
 */
void get_factorials(mpf_t * factorials, int num_factorials){
    int i;
    mpf_t f;
    mpf_init_set_ui(f, 1);
    mpf_init_set_ui(factorials[0], 1);
    for(i = 1; i <= num_factorials; i++){
        mpf_mul_ui(f, f, i);
        mpf_init_set(factorials[i], f);
    }
    mpf_clear(f);
}

/*
 * This method clears the factorials computed and stored in mpf_t * factorials
 */
void clear_factorials(mpf_t * factorials, int num_factorials){
    int i;
    for(i = 0; i <= num_factorials; i++){
        mpf_clear(factorials[i]);
    }
}

/*
 * An iteration of Chudnovsky formula
 */
void Chudnovsky_iteration_v1(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, mpf_t dep_c, 
                        mpf_t dep_d, mpf_t dep_e, mpf_t dividend, mpf_t divisor){
    mpf_mul(dividend, dep_a, dep_e);

    mpf_mul(divisor, dep_b, dep_c);
    mpf_mul(divisor, divisor, dep_d);
    
    mpf_div(dividend, dividend, divisor);

    mpf_add(pi, pi, dividend);
}

/*
 * Sequential Pi number calculation using the Chudnovsky algorithm
 * Single thread implementation
 */
void Chudnovsky_algorithm_v1(mpf_t pi, int num_iterations){
    int num_factorials, i; 
    num_factorials = (num_iterations * 6) + 2;
    mpf_t factorials[num_factorials + 1];
    get_factorials(factorials, num_factorials);   

    mpf_t dep_a, dep_b, dep_c, dep_d, dep_e, e, c, dividend, divisor;
    mpf_inits(dividend, divisor, NULL);
    mpf_init_set_ui(dep_a, 1);
    mpf_init_set_ui(dep_b, 1);
    mpf_init_set_ui(dep_c, 1);
    mpf_init_set_ui(dep_d, 1);
    mpf_init_set_ui(dep_e, A);
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    for(i = 0; i < num_iterations; i ++){
        Chudnovsky_iteration_v1(pi, i, dep_a, dep_b, dep_c, dep_d, dep_e, dividend, divisor);
        //Update dependencies
        mpf_set(dep_a, factorials[6 * (i + 1)]);
        mpf_pow_ui(dep_b, factorials[i + 1], 3);
        mpf_set(dep_c, factorials[3 * (i + 1)]);
        mpf_mul(dep_d, dep_d, c);
        mpf_add_ui(dep_e, dep_e, B);
    }
    mpf_sqrt(e, e);
    mpf_mul_ui(e, e, D);
    mpf_div(pi, e, pi);    
    
    //Clear memory
    clear_factorials(factorials, num_factorials);
    mpf_clears(dep_a, dep_b, dep_c, dep_d, dep_e, c, e, dividend, divisor, NULL);

}
