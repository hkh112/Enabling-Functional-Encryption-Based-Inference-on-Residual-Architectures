import ctypes

typedict = {
    "float": ctypes.c_float,
    "int": ctypes.c_int,
    "double": ctypes.c_double,
    "uint32": ctypes.c_uint32,
    "float*": ctypes.POINTER(ctypes.c_float),
    "int*": ctypes.POINTER(ctypes.c_int),
    "double*": ctypes.POINTER(ctypes.c_double),
    "void": ctypes.c_void_p,
    "uint32*": ctypes.POINTER(ctypes.c_uint32),
}