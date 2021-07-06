#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "BBPAlgorithmV1.h"
#include "BBPAlgorithmV2.h"
#include "BBPAlgorithm.h"
#include "BellardAlgorithm.h"
#include "ChudnovskyAlgorithm.h"


void checkDecimals(mpf_t pi){
    //Cast the number we want to check to string
    int bytes_of_pi = ((pi -> _mp_prec + 1) * sizeof(mp_limb_t)) * sizeof(int); 
    char calculated_pi[bytes_of_pi]; 
    gmp_sprintf(calculated_pi, "%.Ff", pi);

    //Read the correct pi number from numeroPiCorrecto.txt file and compares the decimals to calculated pi
    FILE * file;
    file = fopen("./resources/numeroPiCorrecto.txt", "r");
    if(file == NULL){
        printf("numeroPiCorrecto.txt not found \n");
        exit(-1);
    } 

    char correct_pi_char;
    int i = 0;
    while((correct_pi_char = fgetc(file)) != EOF){
        if( (i >= bytes_of_pi) || (correct_pi_char != calculated_pi[i])){
            break;
        }
        i++;
    }
    i = (i < 2) ? 0: i - 2;
    printf("Match the first %d decimal places \n", i);
    
    fclose(file);
}

void BBPAlgorithmV1(int num_threads, int precision){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int num_iterations = precision;
    
    if(num_threads <= 1){ 
        SequentialBBPAlgorithmV1(pi, num_iterations);
    } else {
        ParallelBBPAlgorithmV1(pi, num_iterations, num_threads);
    }
    
    checkDecimals(pi);
    
    mpf_clear(pi);
}

void BBPAlgorithmV2(int num_threads, int precision){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int num_iterations = precision;
    
    if(num_threads <= 1){ 
        SequentialBBPAlgorithmV2(pi, num_iterations);
    } else {
        ParallelBBPAlgorithmV2(pi, num_iterations, num_threads);
    }
    
    checkDecimals(pi);
    
    mpf_clear(pi);
}

void BBPAlgorithm(int num_threads, int precision){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int num_iterations = precision;
    
    if(num_threads <= 1){ 
        SequentialBBPAlgorithm(pi, num_iterations);
    } else {
        ParallelBBPAlgorithm(pi, num_iterations, num_threads);
    }
    
    checkDecimals(pi);
    
    mpf_clear(pi);
}

void BellardAlgorithm(int num_threads, int precision){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int num_iterations = precision / 3;
    //int num_iterations = (precision + 4 - 1) / 4;  //Division por exceso

    if(num_threads <= 1){ 
        SequentialBellardAlgorithm(pi, num_iterations);
    } else {
        ParallelBellardAlgorithm(pi, num_iterations, num_threads);
    }
    
    checkDecimals(pi);

    mpf_clear(pi);
}   

void ChudnovskyAlgorithm(int num_threads, int precision){
    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_t pi;
    mpf_init_set_ui(pi, 0);
    int num_iterations = (precision + 14 - 1) / 14;  //Division por exceso

    if(num_threads <= 1){ 
        SequentialChudnovskyAlgorithm(pi, num_iterations);
    } else {
        ParallelChudnovskyAlgorithm(pi, num_iterations, num_threads);
    }
    
    checkDecimals(pi);

    mpf_clear(pi);
}   



