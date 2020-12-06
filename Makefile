EXECUTABLE := kmeans
CU_FILES   := kmeans.cu
CU_DEPS    :=
CC_FILES   := cycletimer.cpp


all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64 
CXXFLAGS=-O3 -Wall -g
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc


OBJS=$(OBJDIR)/kmeans.o $(OBJDIR)/cycletimer.o 

.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)


$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)



$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
