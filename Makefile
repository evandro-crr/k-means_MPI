default: all

all:
	mpicc -O3 -o km km.c -lm -g

clean:
	rm km
