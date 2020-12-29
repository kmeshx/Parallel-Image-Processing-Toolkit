CXX = g++ -std=c++11#-std=c++11
OMP = -fopenmp
CFLAGS = -O3 -g -Wall -qopenmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -qopenmp #-qoffload-attribute-target=mic -DRUN_MIC

all: const_thresh

const_thresh: const_thresh.cpp
	            $(ICC) $(ICFLAGS) -o const_thresh const_thresh.cpp $(OMP)

clean:
	rm -f *~ const_thresh