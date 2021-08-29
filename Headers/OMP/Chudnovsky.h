#ifndef CHUDNOVSKY_OMP
#define CHUDNOVSKY_OMP

void Chudnovsky_algorithm_OMP(mpf_t pi, int num_iterations, int num_threads);
void init_dep_a(mpf_t dep_a, int block_start);

#endif

