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
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Chudnovsky formula implementation                                                *
 * This version does not computes all the factorials                                *
 * It implements a single-threaded method and another that can use multiple threads *
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
void ChudnovskyIteration(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, 
                            mpf_t dep_c, mpf_t aux){
    mpf_mul(aux, dep_a, dep_c);
    mpf_div(aux, aux, dep_b);
    
    mpf_add(pi, pi, aux);
}

/*
 * Sequential Pi number calculation using the Chudnovsky algorithm
 * Single thread implementation
 */
void SequentialChudnovskyAlgorithm(mpf_t pi, int num_iterations){
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
        ChudnovskyIteration(pi, i, dep_a, dep_b, dep_c, aux);
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

/*
 * This method is used by ParallelChudnovskyAlgorithm threads
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
 * This method provides an optimal distribution for each thread
 * based on the Chudnovsky iterations analysis.
 * It returns an array of three integers:
 *   distribution[0] -> block size
 *   distribution[1] -> block start
 *   distribution[2] -> block end 
 */
int * getDistribution(int num_threads, int thread_id, int num_iterations){
    int * distribution, i; 
    float work_rates[16][5] = { 59.50,  35.00,  21.35,  15.75,  12.77,
                                40.50,  24.50,	14.35,  10.50,  8.57,
                                0.00,   21.00,	12.25,  9.45,   7.53,
                                0.00,   19.50,	11.55,  8.40,   6.83,
                                0.00,   0.00,	10.85,  7.75,   6.30,
                                0.00,   0.00,	10.30,  7.65,   5.95,
                                0.00,   0.00,	10.00,  7.35,   5.88,
                                0.00,   0.00,	9.35,   7.00,   5.77,
                                0.00,   0.00,	0.00,   6.75,   5.56,
                                0.00,	0.00,	0.00,   6.65,   5.25,
                                0.00,	0.00,	0.00,   6.55,   5.14,
                                0.00,	0.00,	0.00,   6.20,   5.08,
                                0.00,	0.00,	0.00,   0.00,   5.01,
                                0.00,	0.00,	0.00,   0.00,   4.97,
                                0.00,	0.00,	0.00,   0.00,   4.80,
                                0.00,	0.00,	0.00,   0.00,   4.59};

    distribution = malloc(sizeof(int) * 3);
    distribution[0] = work_rates[thread_id][num_threads / 4] * num_iterations / 100;
    distribution[1] = 0;
    for(i = 0; i < thread_id; i ++){
        distribution[1] += work_rates[i][num_threads / 4] * num_iterations / 100;
    }
    distribution[2] = distribution[1] + distribution[0];
    if (thread_id == num_threads -1) distribution[2] = num_iterations;
    
    return distribution;
}

/*
 * Parallel Pi number calculation using the Chudnovsky algorithm
 * Multiple threads can be used
 * The number of iterations is divided by blocks 
 * so each thread calculates a part of pi.  
 */
void ParallelChudnovskyAlgorithm(mpf_t pi, int num_iterations, int num_threads){
    mpf_t e, c;

    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {   
        int thread_id, i, block_size, block_start, block_end, factor_a, * distribution;
        mpf_t local_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;

        thread_id = omp_get_thread_num();
        distribution = getDistribution(num_threads, thread_id, num_iterations);
        block_size = distribution[0];
        block_start = distribution[1];
        block_end = distribution[2];
        
        mpf_init_set_ui(local_pi, 0);    // private thread pi
        mpf_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, aux, NULL);
        initDepA(dep_a, block_start);
        mpf_pow_ui(dep_b, c, block_start);
        mpf_init_set_ui(dep_c, B);
        mpf_mul_ui(dep_c, dep_c, block_start);
        mpf_add_ui(dep_c, dep_c, A);
        factor_a = 12 * block_start;

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = block_start; i < block_end; i++){
                ChudnovskyIteration(local_pi, i, dep_a, dep_b, dep_c, aux);
                //Update dep_a:
                mpf_set_ui(dep_a_dividend, factor_a + 10);
                mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
                mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
                mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);

                mpf_set_ui(dep_a_divisor, i + 1);
                mpf_pow_ui(dep_a_divisor, dep_a_divisor ,3);
                mpf_div(dep_a, dep_a_dividend, dep_a_divisor);
                factor_a += 12;

                //Update dep_b:
                mpf_mul(dep_b, dep_b, c);

                //Update dep_c:
                mpf_add_ui(dep_c, dep_c, B);
            }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(pi, pi, local_pi);
        
        //Clear thread memory
        mpf_clears(local_pi, dep_a, dep_b, dep_c, dep_a_dividend, dep_a_divisor, aux, NULL);   
    }

    mpf_sqrt(e, e);
    mpf_mul_ui(e, e, D);
    mpf_div(pi, e, pi);    
    
    //Clear memory
    mpf_clears(c, e, NULL);
}