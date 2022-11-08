from numba.extending import get_cython_function_address
import ctypes

cpdef api void cprint_double(double d):
    print(d)

cprint_double_addr = get_cython_function_address("sdesolver.util.cprint", "cprint_double")
cprint_double_cfunc = ctypes.CFUNCTYPE(None, ctypes.c_double)(cprint_double_addr)
