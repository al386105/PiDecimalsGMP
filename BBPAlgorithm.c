#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>


void BPP_iteration(mpf_t pi, int n, mpf_t jump, mpf_t m){
    mpf_t a, b, c, d, aux;
    mpf_init_set_ui(a, 4.0);        // (  4 / 8n + 1))
    mpf_init_set_ui(b, 2.0);        // ( -2 / 8n + 4))
    mpf_init_set_ui(c, 1.0);        // ( -1 / 8n + 5))
    mpf_init_set_ui(d, 1.0);        // ( -1 / 8n + 6))
    mpf_init_set_ui(aux, 0);        // (a + b + c + d)  

    int i = n << 3;                 // i = n * 8 
    mpf_div_ui(a, a, i + 1.0);      // a = 4 / (8n + 1)
    mpf_div_ui(b, b, i + 4.0);      // b = 2 / (8n + 4)
    mpf_div_ui(c, c, i + 5.0);      // c = 1 / (8n + 5)
    mpf_div_ui(d, d, i + 6.0);      // d = 1 / (8n + 6)

    // aux = (a - b - c - d)   
    mpf_add(aux, aux, a);
    mpf_sub(aux, aux, b);
    mpf_sub(aux, aux, c);
    mpf_sub(aux, aux, d); 

    // aux = m * aux
    mpf_mul(aux, aux, m);   
    
    mpf_add(pi, pi, aux);    

    // Update m for next iteration: m^n = m^numThreads * m^(n-numThreads) 
    mpf_mul(m, m, jump);  
}

void BBPAlgorithm_SequentialImplementation(mpf_t pi, int numIterations){
   double q = 1.0 / 16.0;
    mpf_t m, quotient;           
    mpf_init_set_ui(m, 1);          // m = (1/16)^n
    mpf_init_set_d(quotient, q);   // quotient = (1/16)      

    int i;
    for(i = 0; i < numIterations; i++){
        BPP_iteration(pi, i, quotient, m);    
    }

}

void BBPAlgorithm_ParallelImplementation(mpf_t pi, int numIterations, int numThreads){
    int myId, i;
    double q = 1.0 / 16.0;
    mpf_t jump, quotient; 
    mpf_init_set_d(quotient, q);            // quotient = (1 / 16)   
    mpf_init_set_ui(jump, 1);        
    mpf_pow_ui(jump, quotient, numThreads); // jump = (1/16)^numThreads

    //Set the number of threads 
    omp_set_num_threads(numThreads);

    #pragma omp parallel private(myId, i)
    {
        myId = omp_get_thread_num();
        mpf_t piLocal, m;
        mpf_init_set_ui(piLocal, 0);    // private thread pi
        mpf_init_set_ui(m, 0);
        mpf_pow_ui(m, quotient, myId);  // m = (1/16)^n          
        
        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = myId; i < numIterations; i+=numThreads){
                BPP_iteration(piLocal, i, jump, m);    
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(pi, pi, piLocal);
        
        //Clear memory
        mpf_clear(piLocal);   
        mpf_clear(m);
                
    }
        
    //Clear memory
    mpf_clear(quotient);
    mpf_clear(jump);
}

