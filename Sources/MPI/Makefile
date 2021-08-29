GCC = mpicc

LIBS = -lgmp

EXECS = parallel.x

default: all

all: $(EXECS)

parallel.x :  PiDecimalsMPI.o PiCalculatorMPI.o ChudnovskyAlgorithmMPI.o ChudnovskyAlgorithmV1MPI.o BellardAlgorithmMPI.o BBPAlgorithmMPI.o OperationsMPI.o
	$(GCC) -fopenmp -o parallel.x PiDecimalsMPI.o PiCalculatorMPI.o ChudnovskyAlgorithmMPI.o ChudnovskyAlgorithmV1MPI.o BellardAlgorithmMPI.o BBPAlgorithmMPI.o OperationsMPI.o $(LIBS)

.c.o:
	$(GCC) -fopenmp -c $*.c

clean:
	rm *.o

clear:
	rm *.o $(EXECS)

# mpicc -fopenmp -o PiDecimalsMPI.x PiDecimalsMPI.c PiCalculatorMPI.c ChudnovskyAlgorithmMPI.c BellardAlgorithmMPI.o BBPAlgorithmMPI.c OperationsMPI.c -lgmp