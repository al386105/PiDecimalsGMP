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
 * This version does not computes all the factorials                                *
 *                                                                                  *
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
void ChudnovskyIteration(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, 
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
void SequentialChudnovskyAlgorithm(mpf_t pi, int num_iterations){
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
        ChudnovskyIteration(pi, i, dep_a, dep_b, dividend, divisor);
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

/*
 * This method is used by ParallelChudnovskyAlgorithmV1 threads
 * for computing the first value of dep_a
 */
void initDepA(mpf_t dep_a, int block_start){
    mpz_t factorial_n, dividend, divisor;
    mpf_t float_dividend, float_divisor;
    mpz_inits(factorial_n, dividend, divisor, NULL);
    mpf_inits(float_dividend, float_divisor, NULL);

    mpz_fac_ui(factorial_n, block_start);
    mpz_fac_ui(divisor, 3 * block_start);
    mpz_fac_ui(dividend, 6 * block_start);

    mpz_pow_ui(factorial_n, factorial_n, 3);
    mpz_mul(divisor, divisor, factorial_n);

    mpf_set_z(float_dividend, dividend);
    mpf_set_z(float_divisor, divisor);

    mpf_div(dep_a, float_dividend, float_divisor);

    mpz_clears(factorial_n, dividend, divisor, NULL);
    mpf_clears(float_dividend, float_divisor, NULL);
}

/*
 * Parallel Pi number calculation using the Chudnovsky algorithm
 * Multiple threads can be used
 * The number of iterations is divided by blocks 
 * so each thread calculates a part of pi.  
 */
void ParallelChudnovskyAlgorithm(mpf_t pi, int num_iterations, int num_threads){
    mpf_t aux, c;
    int block_size;
    
    block_size = (num_iterations + num_threads - 1) / num_threads;
    mpf_init_set_ui(aux, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {   
        int thread_id, i, block_start, block_end, factor_a;
        mpf_t local_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dividend, divisor;

        thread_id = omp_get_thread_num();
        block_start = thread_id * block_size;
        block_end = (thread_id == num_threads - 1) ? num_iterations : block_start + block_size;
        
        mpf_init_set_ui(local_pi, 0);    // private thread pi
        mpf_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, dividend, divisor, NULL);
        initDepA(dep_a, block_start);
        mpf_pow_ui(dep_b, c, block_start);

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = block_start; i < block_end; i++){
                ChudnovskyIteration(local_pi, i, dep_a, dep_b, dividend, divisor);
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

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(pi, pi, local_pi);
        
        //Clear memory
        mpf_clears(local_pi, dep_a, dep_b, dep_a_dividend, dep_a_divisor, dividend, divisor, NULL);   
    }

    mpf_sqrt(aux, aux);
    mpf_mul_ui(aux, aux, D);
    mpf_div(pi, aux, pi);    
    
    //Clear memory
    mpf_clears(c, aux, NULL);
}