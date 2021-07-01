GCC = gcc

LIBS = -lgmp

EXECS = parallel.x

default: all

all: $(EXECS)

parallel.x :  PiDecimals.o PiCalculator.o ChudnovskyAlgorithm.o BBPAlgorithm.o BBPAlgorithmV1.o BBPAlgorithmV2.o
	$(GCC) -fopenmp -o parallel.x PiDecimals.o PiCalculator.o ChudnovskyAlgorithm.o BBPAlgorithm.o BBPAlgorithmV1.o BBPAlgorithmV2.o $(LIBS)

.c.o:
	$(GCC) -fopenmp -c $*.c

clean:
	rm *.o

clear:
	rm *.o $(EXECS)

# gcc -fopenmp -o parallel.x PiDecimals.c PiCalculator.c ChudnovskyAlgorithm.c BBPAlgorithm.c BBPAlgorithmV1.c BBPAlgorithmV2.c -lgmp