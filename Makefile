CXX = g++
CXXFLAGS = -Wall -g -pg -fPIC -O3 #-fsanitize=address -fno-omit-frame-pointer -O0

AR = ar
AR_FLAGS = rvs

SRC_DIR = src
CPPSRC_DIR = $(SRC_DIR)/cpp
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib

CXXLINK =
CXXINCL = /usr/include/eigen3

CXX_SRCS := $(wildcard $(CPPSRC_DIR)/*.cpp)
CXX_OBJS := $(patsubst $(CPPSRC_DIR)/%.cpp,$(BUILD_DIR)/%.o, $(CXX_SRCS))

INC_DIRS := $(CPPSRC_DIR) $(CXXINCL)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CXX += $(CXXFLAGS)

$(BUILD_DIR)/%.o: $(CPPSRC_DIR)/%.cpp $(CPPSRC_DIR)/%.h 
	$(CXX) $(INC_FLAGS) -c -o $@ $<

libbatch.so: $(CXX_OBJS)
	$(CXX) $(CXX_OBJS) -shared -o $(LIB_DIR)/$@

libbatch.a: $(CXX_OBJS)
	$(AR) $(AR_FLAGS) $(LIB_DIR)/$@ $(CXX_OBJS)

dirs:
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(BUILD_DIR)
	
bin: $(CXX_OBJS)
	$(CXX) $(CXXLINK) $^ -o $(BIN_DIR)/$@

clib: dirs libbatch.so libbatch.a 

cython: dirs libbatch.a
	python setup.py build_ext --build-lib $(LIB_DIR) --build-temp $(BUILD_DIR)

lib: dirs clib cython

all: dirs bin lib

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	rm -rf $(LIB_DIR)

print-%  : ; @echo $* = $($*)
