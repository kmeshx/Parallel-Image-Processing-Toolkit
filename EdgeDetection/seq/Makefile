CXX = g++ -std=c++11
OMP = -fopenmp
CFLAGS = -O3 -g -Wall -openmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -openmp -offload-attribute-target=mic -DRUN_MIC

all: edge

wsp: edge.cpp cycletimer.cpp
	            $(ICC) $(ICFLAGS) -o edge cycletimer.cpp edge.cpp $(OMP)

clean:
	rm -f *~ edge