CXX = g++ -std=c++11#-std=c++11
OMP = -fopenmp
CFLAGS = -O3 -g -Wall -qopenmp

ICC = icc -m64
ICFLAGS = -O3 -g -Wall -qopenmp #-qoffload-attribute-target=mic -DRUN_MIC

all: otsu

otsu: otsu.cpp
	            $(ICC) $(ICFLAGS) -o otsu otsu.cpp $(OMP)

clean:
	rm -f *~ otsu