cc = clang
cflags = -Wall -Wextra -Wno-unused-function -fsanitize=address,undefined
metal_flags = -lm -framework Metal -framework Foundation -framework CoreGraphics
opencl_flags = -lm

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    opencl_flags += -framework OpenCL
else
    opencl_flags += -lOpenCL
endif

default:
	@echo "please specify example to build"

metal: metal.m ../ccml.h
	$(cc) $(cflags) $(metal_flags) metal.m -o metal && ./metal
	
metal_debug: metal.m ../ccml.h
	$(cc) $(cflags) -g $(metal_flags) metal.m -o metal_debug && ./metal_debug
	
opencl: opencl.c ../ccml.h
	$(cc) $(cflags) $(opencl_flags) opencl.c -o opencl && ./opencl
	
opencl_debug: opencl.c ../ccml.h
	$(cc) $(cflags) -g $(opencl_flags) opencl.c -o opencl_debug && ./opencl_debug
	
clean:
	@test ! -e ./metal || rm ./metal
	@test ! -e ./metal_debug || rm ./metal_debug
	@test ! -e ./opencl || rm ./opencl
	@test ! -e ./opencl_debug || rm ./opencl_debug
