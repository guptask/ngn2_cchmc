CXX= g++
CXXFLAGS= -c -std=c++11 -Wall -Werror `pkg-config --cflags opencv`
LDFLAGS= `pkg-config --libs opencv`
SRC= src
SOURCES= $(wildcard $(SRC)/*.cpp)
INCLUDIR= $(wildcard $(SRC)/*.hpp)
OBJECTS= $(join $(addsuffix ../, $(dir $(SOURCES))), $(notdir $(SOURCES:.cpp=.o)))

EXECUTABLE = analyze

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	@$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

%.o: $(SRC)/%.cpp $(INCLUDIR)
	@$(CXX) $(CXXFLAGS) $< -o $@

clean:
	@rm -f $(EXECUTABLE) *.o

.PHONY: all clean
