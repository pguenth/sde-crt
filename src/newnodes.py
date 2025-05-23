from grapheval.node import *
from sdesolver import *

import copy

import numpy as np

class SDESolverNode(EvalNode):
    def do(self, parent_data, common, sde, scheme, timestep, observation_times, nthreads=4, supervise=True, particle_count_limit=np.inf, **kwargs):
        solver = SDESolver(scheme)
        return solver.solve(sde, timestep, observation_times, nthreads=nthreads, supervise=supervise, particle_count_limit=particle_count_limit)

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
        print("vnode", self.name, confinements, confine_range) 
        if 'weights' in parent_data:
            weights = copy.copy(parent_data['weights'])

        for conf_idx, conf_cond in confinements:
            points = [p for p in points if conf_cond(p[conf_idx])]
            if 'weights' in parent_data:
                weights = [w for p, w in zip(points, weights) if conf_cond(p[conf_idx])]

        for conf_idx, conf_min, conf_max in confine_range:
            points = [p for p in points if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]
            if 'weights' in parent_data:
                weights = [w for p, w in zip(points, weights) if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]

        if len(points) == 0:
            ret = {'values': np.array([])}
        else:
            ret = {'values' : np.array(points).T[index]}

        if 'weights' in parent_data:
            ret['weights'] = np.array(weights)

        #if index == 1:
        #    # numerical error detection
        #    a = np.count_nonzero(v_array <= 1)
        #    b = np.count_nonzero(v_array <= 0)
        #    if a > 0:
        #        print("decreasing momentum detected in {}/{} particles at node {}".format(a, len(v_array), self.name))
        #    if b > 0:
        #        print("negative momentum detected in {}/{} particles at node {}".format(b, len(v_array), self.name))

        assert len(ret['values']) == len(ret['weights'])
        return ret

#SDEValues('val', PassiveNode('p', points='Test'))
