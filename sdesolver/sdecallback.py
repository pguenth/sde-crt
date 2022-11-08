import inspect
import logging

from collections.abc import Mapping, Sequence
from collections import OrderedDict

from numba.extending import intrinsic
from numba import types, cfunc

from dictproperty.dictproperty import dictproperty

@intrinsic
def address_as_void_pointer(typingctx, src):
    """ 
    returns a void pointer from a given memory address
    source/reference:
        https://stackoverflow.com/a/58561573
        https://stackoverflow.com/q/61509903
    """
    from numba import types 
    from numba.core import cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

def _cfunc_sde_base(types_base, type_return, optional_func=None, param_types=None, **kwargs):
    """
    Generalization of cfunc_coeff and cfunc_boundary to avoid rewriting code.
    """
    
    def deco(func):
        if param_types is None:
            pcount = len(inspect.signature(func).parameters) - len(types_base)
            ptypes = [types.double] * pcount
        else:
            ptypes = param_types

        return cfunc(type_return(*types_base, *ptypes), **kwargs)(func)

    if not optional_func is None:
        r = deco(optional_func)
        return r
    else:
        return deco

def cfunc_coeff(*args, **kwargs):
    """
    a convenience decorator for compiling coefficient functions (drift/diffusion)
    to numba cfuncs callable from C++ code. In any case, the first three arguments
    are `out, t, x`. `t` and `x` are the time and phase-space point at which
    the function should evaluate; `out` is an array where the result should be
    written to. Those three are automatically typed by this decorator.

    The remaining arguments (referred to as 'parameters' further on) are by
    default typed as double when calling this decorator without arguments:

        @cfunc_coeff
        def example(out, t, x, p0, p1, p2):
            pass

    Using arguments, another type can be chosen:

        @cfunc_coeff(param_types=[types.int32, types.int32. types.double])
        def example(out, t, x, an_int, another_int, a_double):
            pass

    Keyword arguments are passed through to numba.cfunc()
    """
    arg_base = (types.CPointer(types.double), types.double, types.CPointer(types.double))
    return _cfunc_sde_base(arg_base, types.void, *args, **kwargs)

def cfunc_boundary(*args, **kwargs):
    """
    a convenience decorator for compiling the boundary function to a numba cfunc
    callable from C++ code. In any case, the first two arguments
    are `t` and `x`, the time and phase-space point at which
    the function should evaluate; those are automatically typed by this 
    decorator. It must return an int stating wether the
    boundary was reached (0: no boundary reached; -1: reserved; other:
    a boundary was reached). The value can be used to differentiate between
    different boundaries.

    The remaining arguments (referred to as 'parameters' further on) are by
    default typed as double when calling this decorator without arguments:

        @cfunc_coeff
        def example(t, x, p0, p1, p2):
            pass

    Using arguments, another type can be chosen:

        @cfunc_coeff(param_types=[types.int32, types.int32. types.double])
        def example(t, x, an_int, another_int, a_double):
            pass

    Keyword arguments are passed through to numba.cfunc()
    """
    arg_base = (types.double, types.CPointer(types.double))
    return _cfunc_sde_base(arg_base, types.int32, *args, **kwargs)

class SDECallbackBase:
    # those must be set by inheritance
    _types_base = None
    _type_return = None
    _skip_cast_types = [types.Array]

    def __init__(self, pyfunc, parameter_types=None, **kwargs):
        """
        A callback which is compiled to a numba-cfunc function and supports
        dynamically setting a variable amount of additional parameters.

        Keyword arguments are passed through to numba.cfunc()

        This serves as a replacement and extension to the cfunc_* decorators.
        """
        self._compile_kwargs = kwargs
        self._pyfunc = pyfunc

        # this is a list of all parameters
        self._pyfunc_param = list(inspect.signature(self._pyfunc).parameters.keys())

        # this is the list of names of the non-obligatory parameters
        self._pyfunc_varparam = self._pyfunc_param[len(self._types_base):]

        self._parameters_dict = None

        self.parameter_types = parameter_types
        self.parameters = None

    ###
    ### Compile cfunc
    ###

    def _recompile_base(self):
        """
        Compile pyfunc to a cfunc, using the parameter types given in
        self.parameter_types for the non-obligatory parameters
        """
        self._cfunc_noparam = cfunc(
                self._type_return(*self._types_base, *self.parameter_types.values()), 
                **self._compile_kwargs
            )(self._pyfunc)
        
    def _recompile_param(self):
        """
        Compile a function that has the non-obligatory parameters fixed to
        the currently set values of self.parameters
        """
        # the lambdas compile only with a tuple
        if self.parameters is None:
            self._cfunc = self._cfunc_noparam
            return

        param_tuple = tuple(self.parameters.values())

        pfunc = self._cfunc_noparam

        if len(self._types_base) == 3:
            self._cfunc = cfunc(self._type_return(*self._types_base))(
                                   lambda out, t, x : pfunc(out, t, x, *param_tuple))
        elif len(self._types_base) == 2:
            self._cfunc = cfunc(self._type_return(*self._types_base))(
                                   lambda t, x : pfunc(t, x, *param_tuple))
        else:
            raise ValueError("types_base must be either 2 or 3 long")

    ###
    ### Import parameters from different data types
    ###


    def _cast_import(self, k, v):
        t = self.parameter_types[k]

        if t in self._skip_cast_types:
            return v

        try:
            return t(v)
        except NotImplementedError:
            logging.warning("Cast not implemented for type {} (of parameter {}). Skipping cast".format(t, k))
            return v

    def _paramdict_from_seq(self, seq):
        """
        import parameters from a sequence type
        return an OrderedDict containing the parameters,
        in order, as key-value pairs.
        """
        if len(seq) < len(self._pyfunc_varparam):
            raise ValueError("seq too short")
        elif len(seq) > len(self._pyfunc_varparam):
            raise ValueError("seq too long")

        m = OrderedDict()
        for k, v in zip(self._pyfunc_varparam, seq):
            m[k] = self._cast_import(k, v)

        return m

    def _paramdict_from_map(self, map_):
        """
        import parameters from a mapping type
        return an OrderedDict containing the parameters,
        in order, as key-value pairs.
        """
        m = OrderedDict()
        for k in self._pyfunc_varparam:
            if not k in map_:
                raise ValueError("parameter {} not given".format(k))

            m[k] = self._cast_import(k, map_[k])

        return m

    ###
    ### Using as decorator
    ###

    def __call__(self, *args):
        """
        required so that using this class as decorator makes sense
        """
        return self._cfunc(*args)

    @classmethod
    def decorator(cls, opt_pyfunc=None, parameter_types=None, **kwargs):
        """ use this for decorator-with-arguments style syntax """
        if not opt_pyfunc is None:
            return cls(opt_pyfunc, parameter_types, **kwargs)

        def deco(funct):
            return cls(funct, parameter_types, **kwargs)

        return deco

    ###
    ### Properties
    ###

    @property
    def pyfunc(self):
        """
        The (original) python callback
        """
        return self._pyfunc

    @property
    def cfunc(self):
        """
        The numba-C-compiled callback, with the parameters fixed if they have
        been set at some point
        """
        return self._cfunc

    @property
    def cfunc_noparameters(self):
        """
        The numba-C-compiled callback, always without fixed parameters
        """
        return self._cfunc_noparameters

    @property
    def address(self):
        """
        the address of the C-compiled function
        """
        return self.cfunc.address

    ###
    ### DictProperties
    ###

    def _parameter_type_set(self, key, ptype):
        """
        Handle parameter types as they are set.
        Up until now, only Array needs special treatment.

        Array: set as immutable since numba assumes globals are immutable
        (see https://github.com/numba/numba/issues/7669#issuecomment-1004750002
         and https://github.com/numba/numba/issues/7723
         for bug reports on this)
        so we have to set the cfunc type signatures to be immutable too so that
        globals can be loaded as parameters. As a side effect, this enforces
        not changing parameters while solving the SDE, and therefore may avoid
        some hard-to-find bugs in the future.
        """
        if type(ptype) is types.Array:
            ptype.mutable = False

        self._parameter_types[key] = ptype

    @dictproperty
    def parameter_types(self):
        """
        OrderedDict containing the C types of the parameters as key-value pairs.
        The types should be numba types
        (https://numba.readthedocs.io/en/stable/reference/types.html)
        and the numba manual recommends to limit yourself to scalar and pointer
        types.

        For every type not given double (float64) is assumed. To give the
        callback access to an array, the following two ways have been tested.
        First, in accordance with the numba recommendations, you can use a
        pointer to your numpy array:

            from numba import types
            import numpy as np
            from sdesolver import SDECallbackCoeff, address_as_void_pointer

            def py_callback(out, t, x, array_ptr, array_len):
                array = carray(address_as_void_pointer(array_ptr), array_len, dtype=np.float64) 
                # work with array

            array_np = np.array([1.0, 2.0], dtype=np.float64) # example array
            cb = SDECallbackCoeff(py_callback)
            cb.parameter_types['array_ptr'] = types.int64
            cb.parameter_types['array_len'] = types.int32
            cb.parameters = {
                'array_ptr' : array_np.ctypes.data,
                'array_len' : len(array_np)
            }

        A much shorter way is to use a memoryview/numba.types.Array type:

            from numba import types
            import numpy as np
            from sdesolver import SDECallbackCoeff

            def py_callback(out, t, x, array):
                # work with array

            array_np = np.array([1.0, 2.0], dtype=np.float64) # example array
            cb = SDECallbackCoeff(py_callback)
            cb.parameter_types['array'] = types.double[:]
            cb.parameters = { 'array' : array_np }

        The latter approach has to make the compiled function signature to
        accept a readonly (immutable) array for numba-internal reasons. There
        are no runtime differences to be expected.
        """
        return self._parameter_types
    
    @parameter_types.setter
    def parameter_types(self, val):
        self._parameter_types = OrderedDict()
        if val is None:
            # default value (all double)
            for k in self._pyfunc_varparam:
                self._parameter_type_set(k, types.double)
        elif isinstance(val, Sequence):
            # list etc. given
            if not len(val) == len(self._pyfunc_varparam):
                raise ValueError("parameter_types must be of the same length as the function parameters (apart from the obligatory ones)")
            for k, v in zip(self._pyfunc_varparam, val):
                self._parameter_type_set(k, v)
        elif isinstance(val, Mapping):
            # dict etc. given
            for k in self._pyfunc_varparam:
                if k in val:
                    self._parameter_type_set(k, val[k])
                else:
                    logging.warning("type for {} is not given. assuming double".format(k))
                    self._parameter_type_set(k, types.double)
        else:
            raise ValueError("parameter_types must be sequence or mapping")

        self._recompile_base()
        self._recompile_param()

    @parameter_types.getitem
    def parameter_types(self, k):
        return self._parameter_types[k]

    @parameter_types.setitem
    def parameter_types(self, k, v):
        self._parameter_type_set(k, v)
        self._recompile_base()
        self._recompile_param()

    @dictproperty
    def parameters(self):
        """
        OrderedDict containing the values of the parameters as key-value pairs.
        Parameters refers to the non-obligatory part of the call signature, i.e.
        for coefficient functions with the signature

            def func(out, t, x, param_a, param_b):
                pass

        the term 'parameters' refers to param_a and param_b in this context.
        """
        return self._parameters_dict

    @parameters.setter
    def parameters(self, d):
        if d is None:
            self._parameters_dict = None
        elif isinstance(d, Sequence):
            self._parameters_dict = self._paramdict_from_seq(d)
        elif isinstance(d, Mapping):
            self._parameters_dict = self._paramdict_from_map(d)
        else:
            raise ValueError("parameters must be either Sequence, Mapping or None")

        self._recompile_param()

    @parameters.getitem
    def parameters(self, k):
        return self._parameters_dict[k]

    @parameters.setitem
    def parameters(self, k, v):
        v_cast = self.parameter_types[k](v)
        self._parameters_dict[k] = v_cast
        self._recompile_param()


class SDECallbackCoeff(SDECallbackBase):
    _types_base = (types.CPointer(types.double), types.double, types.CPointer(types.double))
    _type_return = types.void

class SDECallbackBoundary(SDECallbackBase):
    _types_base = (types.double, types.CPointer(types.double))
    _type_return = types.int32
