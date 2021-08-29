#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include "../../Headers/Sequential/BBP.h"
#include "../../Headers/Sequential/BBP_v1.h"
#include "../../Headers/Sequential/Bellard.h"
#include "../../Headers/Sequential/Chudnovsky_v1.h"
#include "../../Headers/Sequential/Chudnovsky.h"
#include "../../Headers/Common/Check_decimals.h"

double gettimeofday();

void print_running_properties(int precision, int num_iterations){
    printf("  Precision used: %d \n", precision);
    printf("  Iterations done: %d \n", num_iterations);
}

void calculate_Pi(int algorithm, int precision){
    double execution_time;
    struct timeval t1, t2;
    mpf_t pi;
    int num_iterations, decimals_computed;
    
    gettimeofday(&t1, NULL);

    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    mpf_init_set_ui(pi, 0); 
    
    switch (algorithm)
    {
    case 0:
        num_iterations = precision * 0.84;
        printf("  Algorithm: BBP (First version) \n");
        print_running_properties(precision, num_iterations);
        BBP_algorithm_v1(pi, num_iterations);
        break;

    case 1:
        num_iterations = precision * 0.84;
        printf("  Algorithm: BBP (Last version)\n");
        print_running_properties(precision, num_iterations);
        BBP_algorithm(pi, num_iterations);
        break;

    case 2:
        num_iterations = precision / 3;
        printf("  Algorithm: Bellard \n");
        print_running_properties(precision, num_iterations);
        Bellard_algorithm(pi, num_iterations);
        break;
    
    case 3:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        printf("  Algorithm: Chudnovsky  \n");
        print_running_properties(precision, num_iterations);
        Chudnovsky_algorithm_v1(pi, num_iterations);
        break;
    
    case 4:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        printf("  Algorithm: Chudnovsky (Last version) \n");
        print_running_properties(precision, num_iterations);
        Chudnovsky_algorithm(pi, num_iterations);
        break;
    
    default:
        printf("  Algorithm selected is not correct. Try with: \n");
        printf("      algorithm == 0 -> BBP (First version) \n");
        printf("      algorithm == 1 -> BBP (Last version) \n");
        printf("      algorithm == 2 -> Bellard \n");
        printf("      algorithm == 3 -> Chudnovsky (Computing all factorials) \n");
        printf("      algorithm == 4 -> Chudnovsky (Does not compute all factorials) \n");
        printf("\n");
        exit(-1);
        break;
    }

    gettimeofday(&t2, NULL);
    execution_time = ((t2.tv_sec - t1.tv_sec) * 1000000u +  t2.tv_usec - t1.tv_usec)/1.e6; 
    decimals_computed = check_decimals(pi);
    mpf_clear(pi);
    printf("  Match the first %d decimals \n", decimals_computed);
    printf("  Execution time: %f seconds \n", execution_time);
    printf("\n");
}