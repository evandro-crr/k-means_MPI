default: all

all:
	mpicc -O3 -o km km.c -lm

clean:
	rm km
