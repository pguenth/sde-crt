from grapheval.node import EvalNode
import functools
import numpy as np

def _concat_w(w):
    """
    Takes a dict with T as keys and 2d arrays as values.
    The values' 1st dimension's length must be at least
    1 for the largest T, 2 for the second-largest and
    so on.
   

    Consider an input being
    w[0] = [v00, v01, v02] 
    w[1] = [v10, v11, v12] # v12 is unneccessary
    w[2] = [v20, v21, v22] # v21 and v22 are unneccessary
    with vXX being arrays themselves.
    This will then return
    r[0] = np.concatenate((v00))
    r[1] = np.concatenate((v01, v10))
    r[2] = np.concatenate((v02, v11, v20))
    and so on.

    Returns a dict with the same keys, values being
    """
    concats = {}
    for i, _ in enumerate(sorted(w)):
        to_concatenate = []
        for j, (_, wss) in enumerate(sorted(w.items())):
            for k, ws in enumerate(wss):
                if k + j == i:
                    to_concatenate.append(ws)

        concats[T] = np.concatenate(to_concatenate)

    return concats

def _concat_w_np(w):
    """
    Same as _concat_w

    needs w to be an arrays
    """
    w_stacked = np.stack(w)
    indices = np.indices(w_stacked.shape)
    indices_sum = indices[0] + indices[1]
    concats = []
    for i in range(len(w)):
        concats.append(np.concatenate(w_stacked[indices_sum == i]))

    return concats


class GreensFunctionValues(EvalNode):
    """
    calculates an intermediate result that can be used as input to
    InjectionConvolveValues, being independent from the source function.
    Watch out for large cache size.

    Takes a dict as parents with T as keys and ValueNodes as values.
    """
    def do(self, parent_data, common, **kwargs):
        convolved = {}

        values_buffer = np.empty(0)
        weights_buffer = np.empty(0)
        lens = []
        times = sorted(parent_data)

        for i, (T, v_dict) in enumerate(sorted(parent_data.items())):
            convolved[T] = {}
            convolved[T]['values'] = np.concatenate((values_buffer, v_dict['values']))
            values_buffer = convolved[T]['values']
            convolved[T]['weights'] = np.concatenate((weights_buffer, v_dict['weights']))
            weights_buffer = convolved[T]['weights']
            lens.append(len(v_dict['weights']))
            convolved[T]['source_time'] = np.concatenate([l * [times[len(lens) - j - 1]] for j, l in enumerate(lens)])

        return convolved

class InjectionConvolveValues(EvalNode):
    """
    Calculate the impulse response given a source function callback. Requires
    a GreensFunctionValues node as parent. Returns a dict with T as keys and
    the convolved value sets with weights as values (same as a ValueNode).
    Use in HistogramNode the same way you would use a ValueNode.
    """
    def do(self, parent_data, common, source_callback, **kwargs):
        convolved = {}

        for T, v_dict in parent_data.items():
            convolved[T] = {}
            convolved[T]['values'] = v_dict['values']
            convolved[T]['weights'] = v_dict['weights'] * source_callback(v_dict['source_time'])

        return convolved

class InjectionConvolveValuesDirect(EvalNode):
    """
    This node does the exact same as GreensFunctionValues and 
    InjectionConvolveValues together. It has lower memory and cache
    requirements compared to using those two nodes.
    """
    def do(self, parent_data, common, source_callback, **kwargs):
        convolved = {}

        source_values = source_callback(np.array(sorted(parent_data)))
        values_buffer = np.empty(0)
        w_scaled = {}

        # we want the following:
        # convolved_values[T] = concat(values[t] for t < T)
        # convolved_weights[T] = concat(weights[t] * source(T - t) for t < T)
        # which in concrete terms reads
        # convolved_values[0] = values[0]
        # convolved_values[1] = values[0] values[1]
        # convolved_values[2] = values[0] values[1] values[2]
        # and
        # convolved_weights[0] = weights[0]*source(0)
        # convolved_weights[1] = weights[0]*source(1) weights[1]*source(0)
        # convolved_weights[2] = weights[0]*source(2) weights[1]*source(1) weights[2]*source(0)
        # and so on.             |------------------| |------------------| |------------------| 
        # The columns are the arrays w_scaled[0]           w_scaled[1]          w_scaled[2]
        # observing that the sum of the index of w_scaled and the index of its items add up
        # to the same in each row, we use the bottom for loop to calculate the weights.

        # calculating w_scaled and concatenating the values
        for i, (T, v_dict) in enumerate(sorted(parent_data.items())):
            # weights at T multiplied by the source function at each available time
            # i.e. w_scaled[t] = weights[T] * source(t)
            # only calculate for t < (Tmax - T)
            w_scaled[T] = np.outer(source_values[:len(source_values) - i], v_dict['weights'])

            convolved[T] = {}
            convolved[T]['values'] = np.concatenate((values_buffer, v_dict['values']))
            values_buffer = convolved[T]['values']

        # concatenating the weights
        for i, T in enumerate(sorted(parent_data)):
            concats = []
            for j, (t, wss) in enumerate(sorted(w_scaled.items())):
                for k, ws in enumerate(wss):
                    if k + j == i:
                        concats.append(ws)

            convolved[T]['weights'] = np.concatenate(concats)

        return convolved

class MeshgridTimeseriesNode(EvalNode):
    """
    Calculates a meshgrid, i.e. three 2d-arrays, two representing the 
    coordinates T and v, one representing N. Every set of elements of
    the three arrays (with the same index) represents a point on the
    surface, parametrized by the two indizes)

    This supports different v binnings at each time instant.
    """
    def def_kwargs(self, **kwargs):
        return {'plot_kwargs' : {}} | kwargs

    def plot(self, data, ax, common, **kwargs):
        Ts, vs, Ns = data
        return ax.pcolormesh(Ts, vs, np.log(Ns), **(kwargs['plot_kwargs']))

    def do(self, parent_data, common, **kwargs):
        Ts = []
        vs = []
        Ns = []
        for T_, (hist_v, hist_N, *_) in parent_data.items():
            for v_, N_ in zip(hist_v, hist_N):
                Ts.append(T_)
                vs.append(v_)
                Ns.append(N_)

        Ts = np.array(Ts).reshape(len(parent_data), -1)# bin_count)
        vs = np.array(vs).reshape(len(parent_data), -1)# bin_count)
        Ns = np.array(Ns).reshape(len(parent_data), -1)# bin_count)

        return Ts, vs, Ns

class GreensFunction(EvalNode):
    """
    Collects values from a set of ValueNodes (values of the parent dict) given
    for several T (keys of the parent dict) so that they can be convolved with 
    InjectionConvolveHistogram
    """
    def def_kwargs(self, **kwargs):
        return {'plot_kwargs' : {}} | kwargs

    def plot(self, data, ax, common, **kwargs):
        vs, Ts, Ns, _ = data
        mesh_T, mesh_v = np.meshgrid(Ts, vs)
        return ax.pcolormesh(mesh_T, mesh_v, np.log(Ns), **(kwargs['plot_kwargs']))

    def do(self, parent_data, common, **kwargs):
        """
        returns an ndarray with first index being v and second being T
        
        v histograms must have the same bins at all times to allow
        for simple convolution (without interpolation) later.

        also, the histograms must all be normalized with the same normalization
        constant, or better yet not normalized. normalization should be done
        after convolving.
        """
        Ts = []
        vs = None
        v_edges = None
        Ns = []
        for T_, (hist_v, hist_N, err, edges) in sorted(parent_data.items()):
            Ts.append(T_)

            if vs is None:
                vs = hist_v

            if v_edges is None:
                v_edges = edges

            assert len(hist_v) == len(vs) and np.all(hist_v == vs)
            assert len(v_edges) == len(edges) and np.all(v_edges == edges)

            Ns.append(hist_N)

        Ts = np.array(Ts)
        vs = np.array(vs)
        Ns = np.array(Ns).T

        return vs, Ts, Ns, v_edges

class InjectionConvolveHistogram(EvalNode):
    """
    Calculates the convolution given a GreensFunction as parent and a source
    function callback as kwarg.
    """
    def def_kwargs(self, **kwargs):
        return {'plot_kwargs' : {}} | kwargs

    def plot(self, data, ax, common, **kwargs):
        vs, Ts, Ns, _ = data
        mesh_T, mesh_v = np.meshgrid(Ts, vs)
        return ax.pcolormesh(mesh_T, mesh_v, np.log(Ns), **(kwargs['plot_kwargs']))

    def do(self, parent_data, common, **kwargs):
        source = kwargs['source_callback']

        vs, green_T, green_N, v_edges = parent_data
        convolutions = []

        source_values = source(green_T)
        assert len(green_T) == len(source_values)

        for v, N in zip(vs, green_N):
            this_convolution = np.convolve(source_values, N)[:len(green_T)]
            convolutions.append(this_convolution)

        return vs, green_T, np.array(convolutions), v_edges

class ConvolveExtract(EvalNode):
    """
    Base class for ValueExtract and TimeSeriesExtract.
    """
    def def_kwargs(self, **kwargs):
        return {'plot_kwargs' : {},
                'normalize': 'density'
                } | kwargs

    def plot(self, data, ax, common, **kwargs):
        return ax.plot(data[0], data[1], **(kwargs['plot_kwargs']))

    def do(self, parent_data, common, **kwargs):
        vs, Ts, Ns, v_edges = parent_data
        x, N = self._get_extract(vs, Ts, Ns, kwargs)

        if kwargs['normalize'] == 'density':
            db = self._get_db(Ts, v_edges)
            norm = 1 / db / N.sum()
        else:
            norm = 1

        return x, N * norm

class ValueExtract(ConvolveExtract):
    """
    Extract the histogram for the value from a InjectionConvolveHistogram node.
    Give the time as kwarg T. The nearest possible time is selected.
    """
    def _get_db(self, Ts, v_edges):
        return np.diff(v_edges)

    def _get_extract(self, vs, Ts, Ns, kwargs):
        T = kwargs['T']
        idx = np.argmin(np.abs(T - Ts))
        return vs, Ns.T[idx]

class TimeSeriesExtract(ConvolveExtract):
    """
    Same as ValueExtract, but extract the TimeSeries for a given value.
    """
    def _get_extract(self, vs, Ts, Ns, kwargs):
        v = kwargs['v']
        idx = np.argmin(np.abs(v - vs))
        return Ts, Ns[idx]

class Residual(EvalNode):
    def def_kwargs(self, **kwargs):
        return {'plot_kwargs' : {}} | kwargs

    def plot(self, data, ax, common, **kwargs):
        return ax.plot(data[0], data[1], **(kwargs['plot_kwargs']))

    def do(self, parent_data, common, **kwargs):
        (vs0, Ns0), (vs1, Ns1) = parent_data
        assert len(vs0) == len(vs1) and np.all(vs0 == vs1)

        return vs0, (Ns1 - Ns0) / Ns0
