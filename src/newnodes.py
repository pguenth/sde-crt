from grapheval.node import *
from sdesolver import *

import copy

import numpy as np

class PassiveNode(EvalNode):
    """
    A node that returns its kwargs. Use it to inject arbitrary data into
    a chain
    """
    def do(self, parent_data, common, **kwargs):
        return kwargs


class SDESolverNode(EvalNode):
    def do(self, parent_data, common, sde, scheme, timestep, observation_times, nthreads=4, supervise=True, **kwargs):
        solver = SDESolver(scheme)
        return solver.solve(sde, timestep, observation_times, nthreads=nthreads, supervise=supervise)

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
        # filter points
        points = copy.copy(parent_data['x'])

        for conf_idx, conf_cond in confinements:
            points = [p for p in points if conf_cond(p[conf_idx])]

        for conf_idx, conf_min, conf_max in confine_range:
            points = [p for p in points if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]

        if len(points) == 0:
            ret = {'values': np.array([])}
        else:
            ret = {'values' : np.array(points).T[index]}

        # filter weights
        if 'weights' in parent_data:
            weights = copy.copy(parent_data['weights'])
            points = copy.copy(parent_data['x'])
            for conf_idx, conf_cond in confinements:
                weights = [w for p, w in zip(points, weights) if conf_cond(p[conf_idx])]

            for conf_idx, conf_min, conf_max in confine_range:
                weights = [w for p, w in zip(points, weights) if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]

            ret['weights'] = np.array(weights)

        #if index == 1:
        #    # numerical error detection
        #    a = np.count_nonzero(v_array <= 1)
        #    b = np.count_nonzero(v_array <= 0)
        #    if a > 0:
        #        print("decreasing momentum detected in {}/{} particles at node {}".format(a, len(v_array), self.name))
        #    if b > 0:
        #        print("negative momentum detected in {}/{} particles at node {}".format(b, len(v_array), self.name))

        return ret

#SDEValues('val', PassiveNode('p', points='Test'))
