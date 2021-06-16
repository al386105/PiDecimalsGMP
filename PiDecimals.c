#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "PiCalculator.h"

/*************************************************************************************************
                                                                                                
    Compiling the file: gcc -fopenmp PiCalculator.c BPPAlgorithm.c ... -o PiDecimals -lgmp       
    Executing: ./PiDecimals algorithm precision num_threads                                       
        Algorithm can be:                                                                           
        0 -> BBP (Bailey-Borwein-Plouffe)
        1 -> Chudnovsky
 
 *************************************************************************************************/  

double gettimeofday();

int incorrectParams(){
    printf("Parametros introducidos incorrectos. Se debe ejcutar como: \n");
    printf(" ./PiDecimals algorithm precision num_threads \n");
    printf("    Algorithm can be: \n");
    printf("        0 -> BBP (Bailey-Borwein-Plouffe) Version 1 \n");
    printf("        1 -> BBP (Bailey-Borwein-Plouffe) \n");
    printf("        2 -> Chudnovsky \n"); 
}

void piDecimalsTitle(){
    FILE * file;
    file = fopen("./resources/piDecimalsTitle.txt", "r");
    char character;

    int i = 0;
    while((character = fgetc(file)) != EOF){
        printf("%c", character);
    }
      
    fclose(file);
}


int main(int argc, char **argv){    
    int i;
    //Check the number of parameters are correct
    if(argc != 4){
        incorrectParams();
        exit(-1);
    }

    piDecimalsTitle();

    //Take operation, precision, number of iterations and number of threads from params
    int algorithm = atoi(argv[1]);    
    int precision = atoi(argv[2]);
    int num_threads = (atoi(argv[3]) <= 0) ? 1 : atoi(argv[3]);

    //Declare clock variables and start time
    double execution_time;
    struct timeval t1, t2; 
    gettimeofday(&t1, NULL);

    switch (algorithm)
    {
    case 0:
        printf("Algoritmo: BBP (Primera version) \n");
        BBPAlgorithmV1(num_threads, precision); 
        break;
    case 1:
        printf("Algoritmo: BBP \n");
        BBPAlgorithm(num_threads, precision); 
        break;
    case 2:
        printf("Algoritmo: Chudnovsky \n");
        ChudnovskyAlgorithm(num_threads, precision); 
        break;
    default:
        incorrectParams();
        exit(-1);
        break;
    }

    
    gettimeofday(&t2, NULL);
    execution_time = ((t2.tv_sec - t1.tv_sec) * 1000000u +  t2.tv_usec - t1.tv_usec)/1.e6; 
    
    //Print the results
    printf("Precision: %d \n", precision);
    printf("Numero de hebras: %d\n", num_threads);
    printf("Tiempo de ejecucion: %f segundos. \n", execution_time);

    exit(0);
}