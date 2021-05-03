#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "BBPAlgorithm.h"
#include "ChudnovskyAlgorithm.h"

void checkDecimals(mpf_t pi){
    //Cast the number we want to check to string
    int sizeOfCalculatedPi = ((pi -> _mp_prec + 1) * sizeof(mp_limb_t)) * sizeof(int); 
    char calculatedPi[sizeOfCalculatedPi]; 
    gmp_sprintf(calculatedPi, "%.Ff", pi);

    //Read the correct pi number from numeroPiCorrecto.txt file and compares the decimals to calculated pi
    FILE * file;
    file = fopen("./resources/numeroPiCorrecto.txt", "r");
    char correctPiChar;

    int i = 0;
    while((correctPiChar = fgetc(file)) != EOF){
        if( (i >= sizeOfCalculatedPi) || (correctPiChar != calculatedPi[i])){
            break;
        }
        i++;
    }
    i = (i < 2) ? 0: i - 2;
    printf("Coinciden los %d primeros decimales \n", i);
      
    fclose(file);
}

void sequentialPiCalculation(int algorithm, int precision){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int numIterations;
    if(algorithm == 0){ //BBP Algorithm:
        numIterations = precision;
        BBPAlgorithm_SequentialImplementation(pi, numIterations);
    } else {
        numIterations = precision / 14;
        ChudnovskyAlgorithm_SequentialImplementation(pi, numIterations);
    }
    
    checkDecimals(pi);
    
    mpf_clear(pi);
}

void parallelPiCalculation(int algorithm, int precision, int numThreads){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int numIterations;

    if(algorithm == 0){ //BBP Algorithm:
        numIterations = precision;
        BBPAlgorithm_ParallelImplementation(pi, numIterations, numThreads);
    } else {
        numIterations = precision / 14;
        ChudnovskyAlgorithm_ParallelImplementation(pi, numIterations, numThreads);
    }
    
    checkDecimals(pi);

    mpf_clear(pi);
}   

