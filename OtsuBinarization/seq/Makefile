EXECUTABLE := otsu
CU_FILES   := otsu.cu
CU_DEPS    :=
CC_FILES   := ../utils/cycletimer.cpp


all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX = g++ -std=c++11
CXXFLAGS=-O3 -Wall -g


OBJS=$(OBJDIR)/otsu.o $(OBJDIR)/cycletimer.o 

.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)


$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
