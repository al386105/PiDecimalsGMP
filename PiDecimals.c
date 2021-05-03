#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "PiCalculator.h"

/*  Compiling the file: gcc -fopenmp PiCalculator.c BPPAlgorithm.c ... -o PiDecimals -lgmp 
    Executing: ./PiDecimals algorithm precision numThreads
        Algorithm can be:
        0 -> BBP (Bailey-Borwein-Plouffe)
        1 -> Chudnovsky
*/  

int incorrectParams(){
    printf("Parametros introducidos incorrectos. Se debe ejcutar como: \n");
    printf(" ./PiDecimals algorithm precision numThreads \n");
    printf("    Algorithm can be: ");
    printf("        0 -> BBP (Bailey-Borwein-Plouffe) \n");
    printf("        1 -> Chudnovsky \n"); 
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
    int numThreads = atoi(argv[3]);


    //Declare clock variables and start time
    double executionTime;
    struct timeval t1, t2; 
    gettimeofday(&t1, NULL);

    if (numThreads <= 1){
        printf("----IMPLEMENTACION SECUENCIAL----\n");
        sequentialPiCalculation(algorithm, precision); 

    } else {      
        printf("----IMPLEMENTACION PARALELA----\n");   
        parallelPiCalculation(algorithm, precision, numThreads); 
    } 
    
    gettimeofday(&t2, NULL);
    executionTime = ((t2.tv_sec - t1.tv_sec) * 1000000u +  t2.tv_usec - t1.tv_usec)/1.e6; 
    
    //Print the results
    if(algorithm == 0) printf("Algoritmo: BBP \n");
    else printf("Algoritmo: Chudnovsky \n");
    printf("Precision: %d \n", precision);
    printf("Tiempo de ejecucion: %f segundos. \n", executionTime);

    exit(0);
}