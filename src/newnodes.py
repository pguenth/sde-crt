from grapheval.node import *
from sdesolver import *

import numpy as np

class PassiveNode(EvalNode):
    """
    A node that returns its kwargs. Use it to inject arbitrary data into
    a chain
    """
    def do(self, parent_data, common, **kwargs):
        return kwargs


class SDESolverNode(EvalNode):
    def do(self, parent_data, common, sde, scheme, timestep, observation_times, nthreads=4, **kwargs):
        solver = SDESolver(scheme)
        return solver.solve(sde, timestep, observation_times, nthreads=nthreads)

class SDEEscapeFluxNode(EvalNode):
    """
    Filters all pseudo-particles at a given time from a batch.
    boundary_state can be an int or Iterable
    """
    def do(self, parent_data, common, boundary_state, **kwargs):
        sdesolution = parent_data['sdesolver']
        return sdesolution.escaped[boundary_state]

class SDEValuesNode(EvalNode):
    """
    extracts the values of a given dimension (index) from a set of
    pseudo particles
    """

    def do(self, parent_data, common, index, confinements=[], confine_range=[], **kwargs):
        points = parent_data['points'] 
        for conf_idx, conf_cond in confinements:
            points = [p for p in points if conf_cond(p[conf_idx])]

        for conf_idx, conf_min, conf_max in confine_range:
            points = [p for p in points if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]

        v_array = np.array(points).T[index]

        #if index == 1:
        #    # numerical error detection
        #    a = np.count_nonzero(v_array <= 1)
        #    b = np.count_nonzero(v_array <= 0)
        #    if a > 0:
        #        print("decreasing momentum detected in {}/{} particles at node {}".format(a, len(v_array), self.name))
        #    if b > 0:
        #        print("negative momentum detected in {}/{} particles at node {}".format(b, len(v_array), self.name))

        return v_array

#SDEValues('val', PassiveNode('p', points='Test'))