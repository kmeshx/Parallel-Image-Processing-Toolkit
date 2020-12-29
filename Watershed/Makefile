
CXX = g++ -std=c++11
OMP = -fopenmp
CFLAGS = -O3 -g -Wall -openmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -openmp -offload-attribute-target=mic -DRUN_MIC

all: ws

wsp: ws.cpp cycletimer.cpp
	            $(ICC) $(ICFLAGS) -o ws cycletimer.cpp ws.cpp $(OMP)

clean:
	rm -f *~ ws