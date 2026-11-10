CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17
LDFLAGS ?= -luring -lpthread

.PHONY: all clean

all: cpp/search_v3

cpp/search_v3: cpp/search_v3.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f cpp/search_v3
