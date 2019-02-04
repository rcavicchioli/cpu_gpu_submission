TARGET = vkmain
LIBS = -lvulkan
CC = g++
CFLAGS = -g -Wall --std=c++11
INCLUDE = -I/usr/include/vulkan

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) $(CDEFINES) -c $< $(INCLUDE) -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(CDEFINES) $(INCLUDE) -Wall $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)