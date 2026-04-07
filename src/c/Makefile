# Makefile for the C extension module
# Produces fast_gae.so (Linux/macOS) or fast_gae.dll (Windows)

CC      ?= gcc
CFLAGS  := -O3 -march=native -ffast-math -Wall -Wextra -fPIC
LDFLAGS := -lm
SRCS    := fast_gae.c running_stats.c
OBJS    := $(SRCS:.c=.o)

# Detect platform
UNAME := $(shell uname -s 2>/dev/null || echo Windows)

ifeq ($(UNAME), Linux)
  TARGET  := fast_gae.so
  LDFLAGS += -shared
else ifeq ($(UNAME), Darwin)
  TARGET  := fast_gae.so
  LDFLAGS += -shared -dynamiclib
else
  # Windows (MinGW/Cygwin)
  TARGET  := fast_gae.dll
  LDFLAGS += -shared
  CFLAGS  += -DWIN32
endif

.PHONY: all clean test benchmark

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $(TARGET)"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) fast_gae.so fast_gae.dll

test:
	python python_bindings.py

benchmark:
	python python_bindings.py --benchmark
