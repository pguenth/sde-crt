import sys
import numpy as np

from grapheval.node import EvalNode
from grapheval.cache import FileCache

sys.path.insert(0, '../../../lib')
sys.path.insert(0, '../../../../lib')
sys.path.insert(0, '../../lib')
sys.path.insert(0, 'lib')
sys.path.insert(0, '../lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.pybreakpointstate import *
from evaluation.helpers import *
from evaluation.experiment import *

from .basicnodes import *
from .radiativenodes import *


class PhysBatchNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        pyb = kwargs['batch_cls'](phys_params=kwargs['phys_params'], num_params=kwargs['num_params'])
        exset = Experiment(pyb, pyb.batch_params)
        exset.run()

        return exset

    def def_kwargs(self, **kwargs):
        kwargs = {
            'label_fmt_fields' : {},
        } | kwargs

        #if not 'phys_params' in kwargs or not 'num_params' in kwargs:
        #    raise ValueError("phys_params and num_params must be passed")

        return kwargs

    def common(self, common, **kwargs):
        phy = kwargs['phys_params']
        num = kwargs['num_params']
        par = kwargs['batch_cls'].get_batch_params(phy, num)

        return {'batch_param' : par, 'phys_param' : phy, 'num_param' : num, 'label_fmt_fields' : kwargs['label_fmt_fields'] | par | phy | num }

class BatchNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        exset = Experiment(
                    kwargs['batch_cls'], 
                    kwargs['param'],
                )
        exset.run(kwargs['nthreads'])

        return exset

    def def_kwargs(self, **kwargs):
        kwargs = {
            'label_fmt_fields' : {},
            'nthreads' : 1
        } | kwargs

        return kwargs

    def common(self, common, **kwargs):
        return {'batch_param' : kwargs['param'], 'label_fmt_fields' : kwargs['label_fmt_fields']}

class PointNode(EvalNode):
    """
    Retrieves all points of particles from a batch
    """
    def do(self, parent_data, common, end_state=PyBreakpointState.TIME):
        experiment = parent_data['batch']
        a = np.array([p.x for p in experiment.states if p.breakpoint_state == end_state])
        aback = [e for e in a if e[1] <= 1]
        if len(aback) > 0:
            #print("backpropagation detected in {}/{} particles at node {}. printing values: ".format(len(aback), len(a), self.name), aback)
            print("backpropagation detected in {}/{} particles at node {}".format(len(aback), len(a), self.name))
        return a



class ValuesNode(EvalNode):
    """
    Reqiured parents:
        points

    Lambdas in confinements are not pickle-able so there
    is confine_range which takes a list of
        (index, min_value, max_value)
    tuples if you want to use kwargs pickling

    Returns:
        list of all values
    """
    #def def_kwargs(self, **kwargs):
    #    kwargs = {
    #            'confinements': [],
    #            'end_state': PyBreakpointState.TIME
    #        } | kwargs
     
    #    return kwargs

    def do(self, parent_data, common, index, confinements=[], confine_range=[], end_state=PyBreakpointState.TIME, **kwargs):
        points = parent_data['points'] 
        for conf_idx, conf_cond in confinements:
            points = [p for p in points if conf_cond(p[conf_idx])]

        for conf_idx, conf_min, conf_max in confine_range:
            points = [p for p in points if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]

        v_array = np.array([p[index] for p in points])

        if index == 1:
            # numerical error detection
            a = np.count_nonzero(v_array <= 1)
            b = np.count_nonzero(v_array <= 0)
            if a > 0:
                print("decreasing momentum detected in {}/{} particles at node {}".format(a, len(v_array), self.name))
            if b > 0:
                print("negative momentum detected in {}/{} particles at node {}".format(b, len(v_array), self.name))

        return v_array


#class ConfineNode(EvalNode):
#    def do(self, parent_data, common, confinements):
#        points = parent_data['points']
#
#        for conf_idx, conf_cond in confinements:
#            points = [p for p in relevant_pps if conf_cond(p.x[conf_idx])]

        

#class TimelimitWeightsEvalNode(EvalNode):
#    def do(self, parent_data, index, use_integrator=None, confinements=[]):
#        experiment = parent_data['experiment'] 
#
#        if use_integrator is None:
#            weights = np.array([1] * len(experiment.states))
#        else:
#            weights = experiment.integrator_values[use_integrator]
#
#        relevant_states_weights = zip(experiment.states, weights)
#        for conf_idx, conf_cond in confinements:
#            relevant_states_weights = [(p, w) for p, w in zip(experiment.states, weights) if conf_cond(p.x[conf_idx])]
#
#        relevant_weights = [w for p, w in relevant_states_weights if p.breakpoint_state == self.options['end_state']]
#        return relevant_weights


class PointsNodeCache(FileCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extension = ".values"

    @staticmethod
    def _read_file(file):
        pass

    @staticmethod
    def _write_file(file, obj):
        ps = np.array(obj).T[0].T
#        print(ps)

class PointsMergeNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        cc = np.concatenate(parent_data)
        lenstr = str([len(p) for p in parent_data])
        print("collecting points from {} runs having a total of {} particles. the runs have the following particle count: {}".format(len(parent_data), len(cc), lenstr))
        return cc
