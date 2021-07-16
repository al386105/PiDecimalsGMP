GCC = gcc

LIBS = -lgmp

EXECS = parallel.x

default: all

all: $(EXECS)

parallel.x :  PiDecimals.o PiCalculator.o ChudnovskyAlgorithm.o ChudnovskyAlgorithmV1.o BellardAlgorithm.o BBPAlgorithm.o BBPAlgorithmV1.o 
	$(GCC) -fopenmp -o parallel.x PiDecimals.o PiCalculator.o ChudnovskyAlgorithm.o ChudnovskyAlgorithmV1.o BellardAlgorithm.o BBPAlgorithm.o BBPAlgorithmV1.o $(LIBS)

.c.o:
	$(GCC) -fopenmp -c $*.c

clean:
	rm *.o

clear:
	rm *.o $(EXECS)

# gcc -fopenmp -o parallel.x PiDecimals.c PiCalculator.c ChudnovskyAlgorithm.c ChudnovskyAlgorithmV1.c BellardAlgorithm.c BBPAlgorithm.c BBPAlgorithmV1.c -lgmp