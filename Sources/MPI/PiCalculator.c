#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include "mpi.h"
#include "../../Headers/MPI/BBP.h"
#include "../../Headers/MPI/Bellard.h"
#include "../../Headers/MPI/Chudnovsky_v1.h"
#include "../../Headers/MPI/Chudnovsky.h"
#include "../../Headers/Common/Check_decimals.h"

double gettimeofday();


void check_errors_MPI(int num_procs, int precision, int num_iterations, int num_threads, int proc_id, int algorithm){
    if (precision <= 0){
        if(proc_id == 0) printf("  Precision should be greater than cero. \n\n");
        MPI_Finalize();
        exit(-1);
    } 
    if (num_iterations < (num_threads * num_procs)){
        if(proc_id == 0){
            printf("  The number of iterations required for the computation is too small to be solved with %d threads and %d procesess. \n", num_threads, num_procs);
            printf("  Try using a greater precision or lower threads/processes number. \n\n");
        }
        MPI_Finalize();
        exit(-1);
    }
        if (algorithm == 4){ 
            // Last version of Chudnovksy is more efficient when threads and procs are 2 or multiples of four
            if (num_procs > 2 && num_procs % 4 != 0){
                printf("  The last version of Chudnovksy is not eficient with %d procs. \n", num_procs);
                printf("  Try using two nodes or multiples of four (4, 8, 12, 16, ..) \n\n");
                exit(-1);
            } 
    }
}

void print_running_properties_MPI(int num_procs, int precision, int num_iterations, int num_threads){
    printf("  Precision used: %d \n", precision);
    printf("  Iterations done: %d \n", num_iterations);
    printf("  Number of processes: %d\n", num_procs);
    printf("  Number of threads (per process): %d\n", num_threads);
}

void calculate_Pi_MPI(int num_procs, int proc_id, int algorithm, int precision, int num_threads){
    double execution_time;
    struct timeval t1, t2;
    int num_iterations, decimals_computed; 
    mpf_t pi;    

    //Get init time 
    if(proc_id == 0){
        gettimeofday(&t1, NULL);
    }

    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    if (proc_id == 0){
        mpf_init_set_ui(pi, 0);
    }
    
    
    switch (algorithm)
    {
    case 0:
        num_iterations = precision * 0.84;
        check_errors_MPI(num_procs, precision, num_iterations, num_threads, proc_id, algorithm);
        if (proc_id == 0){
            printf("  Algorithm: BBP (Last version)\n");
            print_running_properties_MPI(num_procs, precision, num_iterations, num_threads);
        } 
        BBP_algorithm_MPI(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    case 1:
        num_iterations = precision / 3;
        check_errors_MPI(num_procs, precision, num_iterations, num_threads, proc_id, algorithm);
        if (proc_id == 0){
            printf("  Algorithm: Bellard \n");
            print_running_properties_MPI(num_procs, precision, num_iterations, num_threads);
        } 
        Bellard_algorithm_MPI(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    case 2:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors_MPI(num_procs, precision, num_iterations, num_threads, proc_id, algorithm);
        if (proc_id == 0){
            printf("  Algorithm: Chudnovsky  \n");
            print_running_properties_MPI(num_procs, precision, num_iterations, num_threads);
        } 
        Chudnovsky_algorithm_v1_MPI(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    case 3:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors_MPI(num_procs, precision, num_iterations, num_threads, proc_id, algorithm);
        if (proc_id == 0){
            printf("  Algorithm: Chudnovsky (Without all factorials) \n");
            print_running_properties_MPI(num_procs, precision, num_iterations, num_threads);
        } 
        Chudnovsky_algorithm_MPI(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    default:
        if (proc_id == 0){
            printf("  Algorithm selected is not correct. Try with: \n");
            printf("      algorithm == 0 -> BBP (Last version) \n");
            printf("      algorithm == 1 -> Bellard \n");
            printf("      algorithm == 2 -> Chudnovsky (Computing all factorials) \n");
            printf("      algorithm == 3 -> Chudnovsky (Does not compute all factorials) \n");
            printf("\n");
        } 
        MPI_Finalize();
        exit(-1);
        break;
    }

    //Get time, check decimals, free pi and print the results
    if (proc_id == 0) {  
        gettimeofday(&t2, NULL);
        execution_time = ((t2.tv_sec - t1.tv_sec) * 1000000u +  t2.tv_usec - t1.tv_usec)/1.e6; 
        decimals_computed = check_decimals(pi);
        mpf_clear(pi);
        printf("  Match the first %d decimals. \n", decimals_computed);
        printf("  Execution time: %f seconds. \n", execution_time);
        printf("\n");
    }

}



