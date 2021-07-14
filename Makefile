rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

CXX = g++
CXXFLAGS = -Wall -g -pg -fPIC -O3 #-fsanitize=address -fno-omit-frame-pointer -O0

AR = ar
AR_FLAGS = rvs

DOXY = doxygen
DOXYFILE = Doxyfile
DOXYDIR = docs/doxy

DOCDIR = docs

SRC_DIR = src
CPPSRC_DIR = $(SRC_DIR)/cpp
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib

CXXLINK =
CXXINCL = /usr/include/eigen3

CXX_SRCS := $(call rwildcard,$(CPPSRC_DIR),*.cpp) #$(wildcard $(CPPSRC_DIR)/*.cpp)
CXX_OBJS := $(patsubst $(CPPSRC_DIR)/%.cpp,$(BUILD_DIR)/%.o, $(CXX_SRCS))
CXX_OBJDIRS := $(dir $(CXX_OBJS))

INC_DIRS := $(CPPSRC_DIR) $(CXXINCL)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CXX += $(CXXFLAGS)

$(BUILD_DIR)/%.o: $(CPPSRC_DIR)/%.cpp $(CPPSRC_DIR)/%.h 
	$(CXX) $(INC_FLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(CPPSRC_DIR)/%.cpp
	$(CXX) $(INC_FLAGS) -c -o $@ $<

libbatch.so: $(CXX_OBJS)
	$(CXX) $(CXX_OBJS) -shared -o $(LIB_DIR)/$@

libbatch.a: $(CXX_OBJS)
	$(AR) $(AR_FLAGS) $(LIB_DIR)/$@ $(CXX_OBJS)

dirs:
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(CXX_OBJDIRS)
	
bin: $(CXX_OBJS)
	$(CXX) $(CXXLINK) $^ -o $(BIN_DIR)/$@

clib: dirs libbatch.so libbatch.a 

cython: dirs libbatch.a
	python setup.py build_ext --build-lib $(LIB_DIR) --build-temp $(BUILD_DIR)

lib: dirs clib cython

all: dirs bin lib

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	rm -rf $(LIB_DIR)
	$(MAKE) -C $(DOCDIR) clean

.PHONY: docs
docs: cython
	cd $(DOXYDIR) && $(DOXY) $(DOXYFILE)
	$(MAKE) -C $(DOCDIR) html

print-%  : ; @echo $* = $($*)
