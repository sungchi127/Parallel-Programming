CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -pthread -O3
CXXFLAGS = -Wall -Wextra -pthread -O3
LDLIBS = -lm

all: pi.out

pi.out: pi.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

pi.o: pi.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f pi.o pi.out
