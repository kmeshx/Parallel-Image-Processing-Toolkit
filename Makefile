CXX = g++ -std=c++11#-std=c++11
OMP = -fopenmp -lgomp
CFLAGS = -O3 -g -Wall -openmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -openmp #-offload-attribute-target=mic -DRUN_MIC

all: kmeans

kmeans: kmeans.cpp cycletimer.cpp
	            $(ICC) $(ICFLAGS) -o kmeans cycletimer.cpp kmeans.cpp $(OMP)

clean:
	rm -f *~ kmeans
