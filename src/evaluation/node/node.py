from abc import ABC, abstractmethod, ABCMeta
import logging
from functools import wraps
from itertools import count

def simple_cache(func):
    """
    Caches a function in RAM so that is only executed once
    for its lifetime. Calling it a second time with different
    arguments is NOT leading to executing the function a second
    time, instead the cache is returned, potentially generated with
    different arguments.

    If this functionality is not intended, use functools.cache
    """

    @wraps(func)
    def decorated(*args, **kwargs):
        self = args[0]
        if not '__c_' + func.__name__ in self.__dict__:
            logging.debug("simple_cache (RAM): generating data for {} in {}".format(func.__name__, self.name))
            self.__dict__['__c_' + func.__name__] = func(*args, **kwargs)
        else:
            logging.debug("simple_cache (RAM): using cached data for {} in {}".format(func.__name__, self.name))

        return self.__dict__['__c_' + func.__name__]

    return decorated

def dict_or_list_iter(obj):
    if type(obj) is list:
        return enumerate(obj)
    elif type(obj) is dict:
        return obj.items()
    else:
        raise TypeError("obj must be either dict or list")

class ColorCycle:
    def __init__(self, colors):
        self.colors = colors
        self._next = 0

    def __getitem__(self, subscript):
        if not type(subscript) is int:
            raise ValueError("subscript of a color cycle must be int")

        return self.colors[subscript % len(self.colors)]

    def __next__(self):
        this = self.colors[self._next]
        self._next += 1
        self._next %= len(self.colors)
        return this

# https://stackoverflow.com/a/30019359/2581056
class InstanceCounterMeta(ABCMeta):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._instance_ids = count(0)

class EvalNode(ABC, metaclass=InstanceCounterMeta):
    def __init__(self, name, parents=None, plot=False, cache=None, ignore_cache=False, **kwargs):
        self._id = next(self._instance_ids)
        self.name = name
        self.ext = self.def_kwargs(**kwargs)
        self.cache = cache
        self.ignore_cache = ignore_cache

        # does not propagate to children
        # set to true for inherited classes that are
        # only proving mappings etc. to prevent double caching
        #
        # !!! NOT USED ANYWHERE IN THIS CLASS
        # PLEASE IMPLEMENT IN _DO(...)
        self.never_cache = False

        self.do_plot = plot

        if parents is None:
            parents = {}

        self.parents = None
        self.subclass_init(parents, **self.ext)
        if self.parents is None:
            self.parents = parents

        if type(self.parents) is list:
            self._parent_data = [None] * len(self.parents)
        elif type(self.parents) is dict:
            self._parent_data = {}
        elif isinstance(self.parents, EvalNode):
            self.parents = [self.parents]
            self._parent_data = [None]
        else:
            raise TypeError("Parents must be either list or dict or EvalNode")

    @property
    def id(self):
        return self._id

    def copy(self, name_suffix, plot=None, ignore_cache=None, last_parents=None, memo=None, **kwargs):
        """
        create a copy of the EvalNode and all of its parents.

        If *plot* or *ignore_cache* are not None, the respective values
        are overriden in the copy for this EvalNode and all of its parents.

        The top-level EvalNode (having no parents) is created with
        *last_parents* as parents. Its kwargs are updated with * **kwargs *

        Other EvalNodes in the chain keep the kwargs used at the time
        of their creation.

        If the same instance of EvalNode occurs multiple times in the
        evaluation tree, the first occurence is copied and this copy is
        used on subsequent occurences. This should preserve loops in the
        tree. This is realized with the memo parameter which should be left
        default on manual calls to this method.

        """
        
        if last_parents is None:
            last_parents = {}

        if memo is None:
            memo = {}

        if type(self.parents) is list:
            new_parents = [None] * len(self.parents)
        elif type(self.parents) is dict:
            new_parents = {}

        for n, p in self.parents_iter:
            if p in memo:
                new_parents[n] = memo[p]
            else:
                cpy = p.copy(name_suffix, plot, ignore_cache, last_parents, memo=memo, **kwargs)
                new_parents[n] = cpy
                memo[p] = cpy

        if len(new_parents) == 0:
            new_parents = last_parents
            kwargs_use = self.ext | kwargs
        else:
            kwargs_use = self.ext

        if not self._ax is None:
            new_plot = self._ax
        else:
            new_plot = self._do_plot

        new = type(self)(self.name + name_suffix,
                parents=new_parents,
                plot=new_plot if plot is None else plot,
                cache=self.cache,
                ignore_cache=self.ignore_cache if ignore_cache is None else ignore_cache,
                **kwargs_use
            )

        return new

    def __getitem__(self, index):
        """
        __getitem__ for nodes return a node which in turn 
        subscripts the return value of this nodes do() call 
        on evaluation. (and returns this subscripted value)

        This may be confusing but I found it to be very useful
        in constructing node chains. It also encourages the use
        of nodes as if they were their generated/returned values.

        This also implies that the object returned is kind of
        a "future", only containing actual subscripted data
        when the node chain is run. This also makes the functions
        __contains__ and __iter__ unavailable to base classes
        since it is not known to the base class what subclasses'
        do() call might return.

        You are encouraged to override __contains__ and/or
        __iter__ for custom subclasses if senseful return values
        can be provided. Further information:
            - https://docs.python.org/3/reference/datamodel.html#emulating-container-types
            - https://docs.python.org/3/reference/expressions.html#membership-test-details
        """
        return SubscriptedNode('_{}_subs_{}'.format(self.name, index), parents=self, subscript=index)

    def __contains__(self, item):
        """
        This function always raises an exception. For more
        information on that read __getitem__()
        """
        raise TypeError("Generally, subscripting nodes works by subscripting the return value of do(). Therefore it cannot be known to the base class (EvalNode) if this value contains an item.")

    def __iter__(self, index):
        """
        This function always raises an exception. For more
        information on that read __getitem__()
        """
        raise TypeError("Generally, subscripting nodes works by subscripting the return value of do(). Therefore it cannot be known to the base class (EvalNode) if this value contains an item.")

    @property
    def parents_iter(self):
        """
        Returns an iterator that can be handled uniformly
        for any type of parents (list or dict).
        For lists, it is enumerate(parents), and for dict
        it is parents.items()
        """
        if type(self.parents) is list:
            return enumerate(self.parents)
        elif type(self.parents) is dict:
            return self.parents.items()
        else:
            raise TypeError("parents must be either dict or list")

    def parents_contains(self, key):
        """
        Returns True of the parents iterator contains an item
        with the given key.

        For parents stored as list, this is equal to 'key < len(parents)'
        and for parents stored as dict, this is eqal to 'key in parents'
        """
        if type(self.parents) is list:
            return key < len(self.parents)
        elif type(self.parents) is dict:
            return key in self.parents
        else:
            raise TypeError("parents must be either dict or list")

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, parents):
        self._parents = parents

    @property
    def do_plot(self):
        return self._do_plot

    @do_plot.setter
    def do_plot(self, plot):
        if plot is True or plot is False:
            self._do_plot = plot
            self._ax = None
        else:
            self._do_plot = True
            self._ax = plot


    def __call__(self, ax=None, plot_this=None, memo=None):
        common = {}

        memo = [] if memo is None else memo

        if not plot_this is None:
            self.do_plot = plot_this

        self._do(ax, False, common, memo)

        return memo

    # get data, either from cache or by regenerating
    # if regenerated and a cache is attached to the 
    # eval node, the data is stored
    @simple_cache
    def _generate_v(self, common):
        v = self.do(self._parent_data, common, **self.ext)
        if not self.cache is None:
            self.cache.store(self.name, v)

        return v

    @simple_cache
    def _cacheload_v(self):
        v = self.cache.load(self.name)
        return v

    def _update_common(self, common):
        common_add = self.common(common, **self.ext)
        if not common_add is None:
            common.update(common_add)

    def _do(self, ax, need_data, common, memo):
        # check if a cache exists and if we should use it
        # or if we need data from our parents for regeneration
        if self.cache is None or self.ignore_cache or not self.cache.exists(self.name):
            this_need_data = True
        else:
            this_need_data = False

        self._update_common(common)

        # request data from parents using this info
        must_fill = False # track if any parents return data (not None)
        for k, p in self.parents_iter:
            maybe_data = p._do(ax, this_need_data, common, memo)
            assert not (maybe_data is None and this_need_data) # if we requested data, but recieved None, quit
            self._parent_data[k] = maybe_data
            must_fill = must_fill or not maybe_data is None

        if must_fill:
            # if none or some (but not all) parents returned None (= (>=1 parent returned data))
            # we force those to return us data
            # if there is no cache, we will always land here
            # because we requested data from all parents (this_need_data == True in this case)
            for k, p in self.parents_iter:
                if self._parent_data[k] is None:
                    self._parent_data[k] = p._do(None, True, common, memo)
            v = self._generate_v(common)
            logging.debug("Evaluation chain: _do at {} chose to regenerate the data (1)".format(self.name))
        elif len(self.parents) == 0 and this_need_data:
            # if there are no parents but data is needed
            v = self._generate_v(common)
            logging.debug("Evaluation chain: _do at {} chose to regenerate the data (2)".format(self.name))
        elif need_data:
            # all parents returned None, this means there are no updates
            # in all our ancestors. Since the child calling us needs data
            # we can load the cache
            v = self._cacheload_v()
            logging.debug("Evaluation chain: _do at {} chose to use the cached data".format(self.name))
        else:
            # all parents returned None and the child doesn't need data
            # so we return None to tell it that we don't have updates
            v = None
            logging.debug("Evaluation chain: _do at {} chose to do nothing".format(self.name))

        if self._do_plot and not self in memo:
            memo.append(self)
            plot_on = self._ax if not self._ax is None else ax
            if v is None:
                # if there is need for a plot, but no data available at this point
                # we can use the cache for plotting since there are no updates
                logging.debug("Evaluation chain: _do at {} chose to load the cached data anyways for plotting".format(self.name))
                v = self._cacheload_v()
            else:
                # plot with available data
                logging.debug("Evaluation chain: _do at {} plots with the already available data".format(self.name))

            self.plot(v, plot_on, common, **self.ext)

        return v

    def search_parent(self, starts_with):
        for _, p in self.parents_iter:
            if str(p.name).startswith(starts_with):
                return p

        for _, p in self.parents_iter:
            found_parent = p.search_parent(starts_with) 
            if not found_parent is None:
                return found_parent

        return None

    def def_kwargs(self, **kwargs):
        """
        Override this method if you have default values for optional
        kwargs or other things you want to do to the kwargs.
        Other overriden methods will see the dict returned by this method
        as kwargs.
        """
        return kwargs

    def subclass_init(self, parents, **kwargs):
        """
        Override this method if you need control over how the 
        parents are attached this eval node or if you need
        to modify the parents in some way.

        You need to set self.parents manually in this method

        It is executed at instance __init__
        """
        self.parents = parents

    def common(self, common, **kwargs):
        """
        Override this method to add information to the common dict
        that is shared across this instance and all parents and children
        in the evaluation chain. It is not intended to be used for data that is 
        expensive to calculate or store.

        You can either return a set of keys which are used to update the common
        dict or directly modify the parameter *common* given>

        Called between do and plot.
        """
        pass

    @abstractmethod
    def do(self, parent_data, common, **kwargs):
        """
        Override this method. Perform the required evaluation.
        parent_data is a dict containing the return values of 
        all parent's do calls with the keys as given at creation
        time.

        Do not change the common dict here, because this method
        is not guaranteed to run at every execution of the evaluation
        chain because of caching.
        """
        raise NotImplementedError

    def plot(self, data, ax, common, **kwargs):
        """
        Override this method if this eval node can plot something.
        It is called if requested with the return value from do(..)
        and an axes object to plot on.
        """
        raise NotImplementedError

class SubscriptedNode(EvalNode):
    def subclass_init(self, parents, **kwargs):
        if not isinstance(parents, EvalNode) and len(parents) > 1:
            raise ValueError("SubscriptedNode can only have one parent node")
        self.never_cache = True

    def do(self, parent_data, common, **kwargs):
        return parent_data[0][kwargs['subscript']]

class NodeGroup(EvalNode):
    def subclass_init(self, parents, **kwargs):
        super().subclass_init(parents, **kwargs)
        self.never_cache = True

    def __contains__(self, key):
        return self.parents_contains(key)

    def do(self, parent_data, common, **kwargs):
        return parent_data

class NodePlotGroup(EvalNode):
    def subclass_init(self, parents, **kwargs):
        super().subclass_init(parents, **kwargs)
        self.never_cache = True

    def do(self, parent_data, common, **kwargs):
        return parent_data

    def _do(self, ax, need_data, common, memo):
        # This is basically a part of EvalNode._do, but
        # not accounting for any caching (because this node
        # does no caching) and redirecting one Axes object
        # from ax (which must be an array in this case)
        # to each parent.

        must_fill = False # track if any parents return data (not None)
        for i, (k, p) in enumerate(self.parents_iter):
            maybe_data = p._do(ax[i], need_data, common, memo)
            assert not (maybe_data is None and need_data) # if we requested data, but recieved None, quit
            self._parent_data[k] = maybe_data
            must_fill = must_fill or not maybe_data is None

        if must_fill:
            # if none or some (but not all) parents returned None (= (>=1 parent returned data))
            # we force those to return us data
            # if there is no cache, we will always land here
            # because we requested data from all parents (this_need_data == True in this case)
            for k, p in self.parents_iter:
                if self._parent_data[k] is None:
                    self._parent_data[k] = p._do(None, True, common, memo)

        return self._parent_data



class CommonCallbackNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        return kwargs['callback'](common)

class KwargsCallbackNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        return kwargs['callback'](kwargs)

class NodeSet(EvalNode):
    """
    Creates a set of copies of the eval node tree given parents. 

    You can specify the kwargs given to and the parents 
    of the last EvalNode in the chain for every copy.
    You may also override *plot* and *ignore_cache* for 
    the whole chain.

    ATTENTION: overriding plot and ignore_cache is not
    implemented at the moment and has no effect

    If you specify any subset of these options, the lists 
    must have the same length. This length or the length
    of only one of them that is non-zero determines the
    count of copies.
    """
    def def_kwargs(self, **kwargs):
        kwargs = {
                'kwargs' : [],
                'last_parents' : []
            } | kwargs

        if len(kwargs['kwargs']) == 0:
            kwargs['kwargs'] = [{}] * len(kwargs['last_parents'])
        if len(kwargs['last_parents']) == 0:
            kwargs['last_parents'] = [{}] * len(kwargs['kwargs'])

        return kwargs
        
    def subclass_init(self, parents, **kwargs):
        if len(parents) == 0:
            return
        elif len(parents) == 1:
            common = parents[next(iter(parents))]
        elif len(parents) != 1:
            common = NodeGroup('g', parents)

        this_parents = {}
        for i, (lp, kw) in enumerate(zip(kwargs['last_parents'], kwargs['kwargs'])):
            cpy = common.copy(str(i), last_parents=lp, **kw)
            this_parents[i] = cpy
        
        self.parents = this_parents

    def do(self, parent_data, common, **kwargs):
        return parent_data

class NodeSetItem(EvalNode):
    """
    Gets one item with key from a NodeSet
    """
    def subclass_init(self, parents, **kwargs):
        super().subclass_init(parents, **kwargs)
        self.never_cache = True

    def do(self, parent_data, common, **kwargs):
        node_set = parent_data['set']
        return node_set[kwargs['key']]

class DebugOutNode(EvalNode):
    def subclass_init(self, parents, **kwargs):
        if not isinstance(parents, EvalNode) and len(parents) > 1:
            raise ValueError("DebugOutNode can only have one parent node")

        self.never_cache = True

        if isinstance(parents, EvalNode):
            self.ext['parent_name'] = parents.name
        else:
            self.ext['parent_name'] = parents[0].name

        self._do_plot = True

    def do(self, parent_data, common, **kwargs):
        return parent_data[0]

    def plot(self, data, ax, common, **kwargs):
        print(kwargs['parent_name'], data, common)

