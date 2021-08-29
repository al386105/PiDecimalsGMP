CC = gcc

LIBS = -lgmp

EXECS = parallel.x

default: all

all: $(EXECS)

parallel.x :  PiDecimals.o PiCalculator.o BBP_v1.o BBP.o Bellard.o Chudnovsky_v1.o Chudnovsky.o
	$(CC) -o parallel.x PiDecimals.o PiCalculator.o BBP_v1.o BBP.o Bellard.o Chudnovsky_v1.o Chudnovsky.o $(LIBS)

.c.o:
	$(CC) -c $*.c

clean:
	rm *.o

clear:
	rm *.o $(EXECS)
