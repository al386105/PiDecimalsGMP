#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

void getFactorials(mpf_t * factorials, int numFactorials){
    int i;
    mpf_t f;
    mpf_init_set_ui(f, 1);
    mpf_init_set_ui(factorials[0], 1);
    for(i = 1; i <= numFactorials; i++){
        mpf_mul_ui(f, f, i);
        mpf_init_set(factorials[i], f);
    }
    mpf_clear(f);
    gmp_printf("SOY GET FACTORIALS \n"); 

}

void clearFactorials(mpf_t * factorials, int numFactorials){
    int i;
    for(i = 0; i <= numFactorials; i++){
        mpf_clear(factorials[i]);
    }
}

void Chudnovsky_iteration(mpf_t pi, int n, mpf_t depA, mpf_t depB, mpf_t depC, mpf_t depD){
    mpf_t dividend, divisor;
    mpf_init_set_ui(dividend, n);
    mpf_mul_ui(dividend, dividend, 545140134);
    mpf_add_ui(dividend, dividend, 13591409);
    mpf_mul(dividend, dividend, depA);

    mpf_init_set_ui(divisor, 0);
    mpf_mul(divisor, depB, depC);
    mpf_mul(divisor, divisor, depD);

    mpf_div(dividend, dividend, divisor);
    gmp_printf("++ %F.f \n ", pi);
    mpf_add(pi, pi, dividend);

}

void ChudnovskyAlgorithm_SequentialImplementation(mpf_t pi, int precision, int numIterations){
    int numFactorials, i; 
    numFactorials = numIterations * 6;
    mpf_t factorials[numFactorials];
    gmp_printf("MI PRINT 1, PI = %F.f \n ",pi); //ESTE SI FUNCIONA
    getFactorials(factorials, numFactorials);   // LLAMADA AL METODO FACTORIALS
    gmp_printf("MI PRINT 2, PI = %F.f \n ",pi); //ESTE NO FUNCIONA

    mpf_t depA, depB, depC, depD, aux, c;
    mpf_init_set_ui(depA, 1);
    mpf_init_set_ui(depB, 1);
    mpf_init_set_ui(depC, 1);
    mpf_init_set_ui(depD, 1);
    mpf_init_set_ui(aux, 10005);
    mpf_init_set_ui(c, 640320);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);
    for(i = 0; i < numIterations; i ++){
        Chudnovsky_iteration(pi, i, depA, depB, depC, depD);
        //Update dependencies
        //mpf_set(depA, factorials[6 * (i + 1)]);
        //mpf_pow_ui(depB, factorials[i + 1], 3);
        //mpf_set(depC, factorials[3 * (i + 1)]);
        //mpf_mul(depD, depD, c);        
    }
    //mpf_sqrt(aux, aux);
    //mpf_mul_ui(aux, aux, 426880);
    //mpf_div(pi, aux, pi);    
    //clearFactorials(factorials, numFactorials);

}

void ChudnovskyAlgorithm_ParallelImplementation(mpf_t pi, int numIterations, int numThreads){
    int numFactorials = numIterations * 6;
    mpf_t factorials[numFactorials];
    getFactorials(factorials, numFactorials);
}

