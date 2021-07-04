#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005

/*Chudnovsky formula
*************************************************************************************
*     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)              * 
*    --------------------  = SUMMATORY( ----------------------------- ),  n >=0     *
*            pi                            (n!)^3 (3n)! (-640320)^3n                *
*************************************************************************************/

void getFactorials(mpf_t * factorials, int num_factorials){
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

void clearFactorials(mpf_t * factorials, int num_factorials){
    int i;
    for(i = 0; i <= num_factorials; i++){
        mpf_clear(factorials[i]);
    }
}

void ChudnovskyIteration(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, mpf_t dep_c, mpf_t dep_d, 
                                mpf_t dividend, mpf_t divisor){
    mpf_set_ui(dividend, n);
    mpf_mul_ui(dividend, dividend, B);
    mpf_add_ui(dividend, dividend, A);
    mpf_mul(dividend, dividend, dep_a);

    mpf_set_ui(divisor, 0);
    mpf_mul(divisor, dep_b, dep_c);
    mpf_mul(divisor, divisor, dep_d);

    mpf_div(dividend, dividend, divisor);

    mpf_add(pi, pi, dividend);
}

void SequentialChudnovskyAlgorithm(mpf_t pi, int num_iterations){
    int num_factorials, i; 
    num_factorials = num_iterations * 6;
    mpf_t factorials[num_factorials + 1];
    getFactorials(factorials, num_factorials);   

    mpf_t dep_a, dep_b, dep_c, dep_d, aux, c, dividend, divisor;
    mpf_inits(dividend, divisor, NULL);
    mpf_init_set_ui(dep_a, 1);
    mpf_init_set_ui(dep_b, 1);
    mpf_init_set_ui(dep_c, 1);
    mpf_init_set_ui(dep_d, 1);
    mpf_init_set_ui(aux, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    for(i = 0; i < num_iterations; i ++){
        ChudnovskyIteration(pi, i, dep_a, dep_b, dep_c, dep_d, dividend, divisor);
        //Update dependencies
        mpf_set(dep_a, factorials[6 * (i + 1)]);
        mpf_pow_ui(dep_b, factorials[i + 1], 3);
        mpf_set(dep_c, factorials[3 * (i + 1)]);
        mpf_mul(dep_d, dep_d, c);
    }
    mpf_sqrt(aux, aux);
    mpf_mul_ui(aux, aux, D);
    mpf_div(pi, aux, pi);    
    
    //Clear memory
    clearFactorials(factorials, num_factorials);
    mpf_clears(dep_a, dep_b, dep_c, dep_d, c, aux, dividend, divisor, NULL);

}

void ParallelChudnovskyAlgorithm(mpf_t pi, int num_iterations, int num_threads){
    mpf_t aux, c;
    int num_factorials, block_size;
    
    num_factorials = num_iterations * 6;
    mpf_t factorials[num_factorials + 1];
    getFactorials(factorials, num_factorials);

    block_size = (num_iterations + num_threads - 1) / num_threads;
    mpf_init_set_ui(aux, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {   
        int thread_id, i, block_start, block_end;
        mpf_t local_pi, dep_a, dep_b, dep_c, dep_d, dividend, divisor;


        thread_id = omp_get_thread_num();
        block_start = thread_id * block_size;
        block_end = (thread_id == num_threads - 1) ? num_iterations : block_start + block_size;
        
        mpf_init_set_ui(local_pi, 0);    // private thread pi
        mpf_inits(dividend, divisor, NULL);
        mpf_init_set(dep_a, factorials[block_start * 6]);
        mpf_init_set(dep_b, factorials[block_start]);
        mpf_pow_ui(dep_b, dep_b, 3);
        mpf_init_set(dep_c, factorials[block_start * 3]);
        mpf_init_set_ui(dep_d, C);
        mpf_neg(dep_d, dep_d);
        mpf_pow_ui(dep_d, dep_d, block_start * 3);

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = block_start; i < block_end; i++){
                ChudnovskyIteration(local_pi, i, dep_a, dep_b, dep_c, dep_d, dividend, divisor);
                //Update dependencies
                mpf_set(dep_a, factorials[6 * (i + 1)]);
                mpf_pow_ui(dep_b, factorials[i + 1], 3);
                mpf_set(dep_c, factorials[3 * (i + 1)]);
                mpf_mul(dep_d, dep_d, c);
            }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(pi, pi, local_pi);
        
        //Clear memory
        mpf_clears(local_pi, dep_a, dep_b, dep_c, dep_d, dividend, divisor, NULL);   
    }

    mpf_sqrt(aux, aux);
    mpf_mul_ui(aux, aux, D);
    mpf_div(pi, aux, pi);    
    
    //Clear memory
    clearFactorials(factorials, num_factorials);
    mpf_clears(c, aux, NULL);
}

