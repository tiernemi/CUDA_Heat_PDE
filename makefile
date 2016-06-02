# Compilers and commands
CC=	gcc
CXX= g++
NVCC= nvcc
LINK= nvcc
DEL_FILE= rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -lcuda $(PARALLEL) $(DEFINES)
CXXFLAGS    = -lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall -lcuda $(PARALLEL) $(DEFINES) -std=c++11

NVCCFLAGS	= -O5 -DWITH_MY_DEBUG -std=c++11 -arch=sm_21 --relocatable-device-code true -lcudadevrt --use_fast_math 
LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SRC=cuda_sim.cu main.cpp radiator.cpp
OBJ=radiator.o main.o  cuda_sim.o
SOURCES=$(SRC)
OBJECTS=$(OBJ)
TARGET= rad

all: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) -lcudadevrt

main.o: main.cpp
	$(CXX) -c $< $(CXXFLAGS) -I$(INCPATH)

radiator.o: radiator.cpp radiator.hpp
	$(CXX) -c $< $(CXXFLAGS) -I$(INCPATH)

# seperate compilation
cuda_sim.o: cuda_sim.cu cuda_sim.hpp
	$(NVCC) -c $< $(NVCCFLAGS) -I$(INCPATH)

#%.o: %.c
#	$(NVCC) $< -c $(NVCCFLAGS) -I$(INCPATH)

#%.o: %.cu
#	$(NVCC) $< -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
