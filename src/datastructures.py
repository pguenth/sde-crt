import numpy as np
from collections.abc import MutableSequence

class AppendableNumpyArray:
    def __init__(self, *args, **kwargs):
        self._nparray = np.array(*args, **kwargs)
        self._list = None

        self._list_outdated = True
        self._nparray_outdated = False

    @property
    def _as_list(self):
        if self._list is None or self._list_outdated is True:
            self._list = list(self._nparray)

        # because of mutable access
        self._nparray_outdated = True

        return self._list

    @property
    def _as_nparray(self):
        if self._nparray is None or self._nparray_outdated is True:
            self._nparray = np.array(self._list)

        # because of mutable access
        self._list_outdated = True

        return self._nparray

    def __getitem__(self, s):
        return self._as_nparray[s]

    def __len__(self):
        return len(self._as_nparray)

    def __setitem__(self, s, value):
        self._list_outdated = True
        self._as_nparray[s] = value





class SDEPseudoParticleSet(MutableSequence):
    def __init__(self, rtype=np.array):
        self._t_list = []
        self._x_list = []
        self.rtype = rtype
        self.set_type = rtype

    @property
    def t(self):
        # mutability?
        return self.rtype(self._t_list)

    @property
    def x(self):
        # mutability?
        return self.rtype(self._x_list)

    def __getitem__(self, s):
        return self._t_list[s], self._x_list[s]

    def __len__(self):
        return len(self._t_list)

    def __setitem__(self, idx, tx):
        t, x = tx
        self._t_list[idx] = t
        self._x_list[idx] = self.set_type(x)

    def __delitem__(self, idx):
        del self._t_list[idx]
        del self._x_list[idx]

    def insert(self, idx, tx):
        t, x = tx
        self._t_list.insert(idx, t)
        self._x_list.insert(idx, self.set_type(x))

    def append(self, t_or_tx, x=None):
        if x is None:
            t, x = t_or_tx
        else:
            t = t_or_tx
               
        self._t_list.append(t)
        self._x_list.append(self.set_type(x))


class SDEPseudoParticle:
    finished_reason : str
    def __init__(self, t0, x0, finished=False, finished_reason='none'):
        self.t = t0
        self.x = x0
        self.finished = finished
        self.finished_reason = finished_reason

    def __deepcopy__(self, memo):
        return self.copy()

    def copy(self):
        return SDEPseudoParticle(self.t, self.x, self.finished, self.finished_reason)


class SDEPPStateOldstyle:
    """ this class is for bodging this solver together with the C++ solver """
    def __init__(self, t, x, breakpoint_state):
        from pybatch.pybreakpointstate import PyBreakpointState

        self.t = t
        self.x = x

        if breakpoint_state == 0:
            self.breakpoint_state = PyBreakpointState.TIME
        else:
            self.breakpoint_state = PyBreakpointState.NONE

