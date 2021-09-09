#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include "../../Headers/OMP/BBP.h"
#include "../../Headers/OMP/BBP_v1.h"
#include "../../Headers/OMP/Bellard.h"
#include "../../Headers/OMP/Chudnovsky_v1.h"
#include "../../Headers/OMP/Chudnovsky.h"
#include "../../Headers/Common/Check_decimals.h"

double gettimeofday();

void check_errors_OMP(int precision, int num_iterations, int num_threads, int algorithm){
    if (precision <= 0){
        printf("  Precision should be greater than cero. \n\n");
        exit(-1);
    } 
    if (num_iterations < num_threads){
        printf("  The number of iterations required for the computation is too small to be solved with %d threads. \n", num_threads);
        printf("  Try using a greater precision or lower threads number. \n\n");
        exit(-1);
    }
    if (algorithm == 4){ 
        // Last version of Chudnovksy is more efficient when threads and procs are 2 or multiples of four
        if (num_threads > 2 && num_threads % 4 != 0){
            printf("  The last version of Chudnovksy is not eficient with %d threads. \n", num_threads);
            printf("  Try using two threads or multiples of four (4, 8, 12, 16, ..) \n\n");
            exit(-1);
        } 
    }
}

void print_running_properties_OMP(int precision, int num_iterations, int num_threads){
    printf("  Precision used: %d \n", precision);
    printf("  Iterations done: %d \n", num_iterations);
    printf("  Number of threads: %d\n", num_threads);
}

void calculate_Pi_OMP(int algorithm, int precision, int num_threads){
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
        check_errors_OMP(precision, num_iterations, num_threads, algorithm);
        printf("  Algorithm: BBP (First version) \n");
        print_running_properties_OMP(precision, num_iterations, num_threads);
        BBP_algorithm_v1_OMP(pi, num_iterations, num_threads);
        break;

    case 1:
        num_iterations = precision * 0.84;
        check_errors_OMP(precision, num_iterations, num_threads, algorithm);
        printf("  Algorithm: BBP (Last version)\n");
        print_running_properties_OMP(precision, num_iterations, num_threads);      
        BBP_algorithm_OMP(pi, num_iterations, num_threads);
        break;

    case 2:
        num_iterations = precision / 3;
        check_errors_OMP(precision, num_iterations, num_threads, algorithm);
        printf("  Algorithm: Bellard \n");
        print_running_properties_OMP(precision, num_iterations, num_threads);
        Bellard_algorithm_OMP(pi, num_iterations, num_threads);
        break;

    case 3:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors_OMP(precision, num_iterations, num_threads, algorithm);
        printf("  Algorithm: Chudnovsky  \n");
        print_running_properties_OMP(precision, num_iterations, num_threads);
        Chudnovsky_algorithm_v1_OMP(pi, num_iterations, num_threads);
        break;

    case 4:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors_OMP(precision, num_iterations, num_threads, algorithm);
        printf("  Algorithm: Chudnovsky (Last version) \n");
        print_running_properties_OMP(precision, num_iterations, num_threads);
        Chudnovsky_algorithm_OMP(pi, num_iterations, num_threads);
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
    printf("  Match the first %d decimals. \n", decimals_computed);
    printf("  Execution time: %f seconds. \n", execution_time);
    printf("\n");
}