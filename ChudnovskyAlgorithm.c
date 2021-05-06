#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

/*Chudnovsky formula
*************************************************************************************
*     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)              * 
*    --------------------  = SUMMATORY( ----------------------------- ),  n >=0     *
*            pi                            (n!)^3 (3n)! (-640320)^3n                *
*************************************************************************************/

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
}

void clearFactorials(mpf_t * factorials, int numFactorials){
    int i;
    for(i = 0; i <= numFactorials; i++){
        mpf_clear(factorials[i]);
    }
}

void ChudnovskyIteration(mpf_t pi, int n, mpf_t depA, mpf_t depB, mpf_t depC, mpf_t depD){
    mpf_t dividend, divisor;
    mpf_init_set_ui(dividend, n);
    mpf_mul_ui(dividend, dividend, 545140134);
    mpf_add_ui(dividend, dividend, 13591409);
    mpf_mul(dividend, dividend, depA);

    mpf_init_set_ui(divisor, 0);
    mpf_mul(divisor, depB, depC);
    mpf_mul(divisor, divisor, depD);

    mpf_div(dividend, dividend, divisor);

    mpf_add(pi, pi, dividend);
}

void ChudnovskyAlgorithmSequentialImplementation(mpf_t pi, int numIterations){
    int numFactorials, i; 
    numFactorials = numIterations * 6;
    mpf_t factorials[numFactorials + 1];
    getFactorials(factorials, numFactorials);   

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
        ChudnovskyIteration(pi, i, depA, depB, depC, depD);
        //Update dependencies
        mpf_set(depA, factorials[6 * (i + 1)]);
        mpf_pow_ui(depB, factorials[i + 1], 3);
        mpf_set(depC, factorials[3 * (i + 1)]);
        mpf_mul(depD, depD, c);
    }
    mpf_sqrt(aux, aux);
    mpf_mul_ui(aux, aux, 426880);
    mpf_div(pi, aux, pi);    
    
    //Free memory
    clearFactorials(factorials, numFactorials);
    mpf_clear(depA); mpf_clear(depB); mpf_clear(depC); mpf_clear(depD); mpf_clear(c); mpf_clear(aux);
}

void ChudnovskyAlgorithmParallelImplementation(mpf_t pi, int numIterations, int numThreads){
    int numFactorials, i, myId, blockSize, blockStart, blockEnd;
    mpf_t piLocal, depA, depB, depC, depD, c, aux;
    
    numFactorials = numIterations * 6;
    mpf_t factorials[numFactorials + 1];
    getFactorials(factorials, numFactorials);

    blockSize = (numIterations + numThreads - 1) / numThreads;
    mpf_init_set_ui(aux, 10005);
    mpf_init_set_ui(c, 640320);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(numThreads);

    #pragma omp parallel private(myId, i, blockStart, blockEnd, piLocal, depA, depB, depC, depD)
    {
        myId = omp_get_thread_num();
        blockStart = myId * blockSize;
        blockEnd = (myId == numThreads - 1) ? numIterations : blockStart + blockSize;
        
        mpf_init_set_ui(piLocal, 0);    // private thread pi
        mpf_init_set(depA, factorials[blockStart * 6]);
        mpf_init_set(depB, factorials[blockStart]);
        mpf_pow_ui(depB, depB, 3);
        mpf_init_set(depC, factorials[blockStart * 3]);
        mpf_init_set_ui(depD, 640320);
        mpf_neg(depD, depD);
        mpf_pow_ui(depD, depD, blockStart * 3);

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = blockStart; i < blockEnd; i++){
                ChudnovskyIteration(piLocal, i, depA, depB, depC, depD);
                //Update dependencies
                mpf_set(depA, factorials[6 * (i + 1)]);
                mpf_pow_ui(depB, factorials[i + 1], 3);
                mpf_set(depC, factorials[3 * (i + 1)]);
                mpf_mul(depD, depD, c);
            }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(pi, pi, piLocal);
        
        //Free memory
        mpf_clear(piLocal); mpf_clear(depA); mpf_clear(depB); mpf_clear(depC); mpf_clear(depD);  
    }

    mpf_sqrt(aux, aux);
    mpf_mul_ui(aux, aux, 426880);
    mpf_div(pi, aux, pi);    
    
    //Free memory
    clearFactorials(factorials, numFactorials);
    mpf_clear(c); mpf_clear(aux);
}

