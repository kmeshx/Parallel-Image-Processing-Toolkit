CXX = g++ -std=c++11#-std=c++11
OMP = -fopenmp -lgomp
CFLAGS = -O3 -g -Wall -qopenmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -qopenmp #-offload-attribute-target=mic -DRUN_MIC

all: kmeans

kmeans: kmeans.cpp ../../utils/cycletimer.cpp
	            $(ICC) $(ICFLAGS) -o kmeans ../../utils/cycletimer.cpp kmeans.cpp $(OMP)

clean:
	rm -f *~ kmeans
