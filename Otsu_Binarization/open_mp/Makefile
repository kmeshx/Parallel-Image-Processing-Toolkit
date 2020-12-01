CXX = g++ -std=c++11
OMP = -fopenmp
CFLAGS = -O3 -g -Wall -qopenmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -qopenmp #-offload-attribute-target=mic -DRUN_MIC

all: otsu

otsu: otsu.cpp cycletimer.cpp
	            $(ICC) $(ICFLAGS) -o otsu cycletimer.cpp otsu.cpp $(OMP)

clean:
	rm -f *~ otsu
