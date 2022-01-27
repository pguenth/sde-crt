from abc import ABC, abstractmethod, ABCMeta
from collections.abc import MutableMapping
import copy
import logging
from functools import wraps
from itertools import count
import matplotlib.axes

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
    _base_id = count(0)

    def __init__(self, name, parents=None, plot=False, cache=None, ignore_cache=False, **kwargs):
        """
        This is the base class for everything in a node chain.
        Every decendant class of this class should implement one (atomic) step
        or operation in the process of evaluating an experiment. Then you can
        put together several instances of EvalNode to form an evaluation chain.

        Decendants should override the following abstract methods:
         * do: Must be overriden. This is the entrypoint for performing the 
               neccessary calculations etc.
         * plot: Optional. Show the data on a specific axis object.
         * def_kwargs: Optional. Set defaults for subclass-specific kwargs.
         * common: Optional. Add values to a dictionary shared across the chain.
         * subclass_init: Optional. Perform initialisation of neccessary. (Do not
                override __init__()
         * __contains__: Optional. As specified in the python object model docs.
                Return wether a key is contained in the return value of do().
        """
        self._id = next(self._instance_ids)
        self._global_id = next(self._base_id)
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

        self.plot_on = plot

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

    @property
    def global_id(self):
        return self._global_id

    def copy(self, name_suffix, plot=None, ignore_cache=None, last_parents=None, last_kwargs=None, memo=None):#, **kwargs):
        """
        create a copy of the EvalNode and all of its parents.

        If *plot* or *ignore_cache* are not None, the respective values
        are overriden in the copy for this EvalNode and all of its parents.

        The top-level EvalNode (having no parents) is created with
        *last_parents* as parents. Its kwargs are updated with *last_kwargs*
        where kwargs which are itself dicts are merged into the existing
        dicts. The **kwargs of the EvalNode instance that is copied is
        deepcopied, which may lead to unexpected behaviour. (Maybe add:
        optionally deepcopy or copy)

        Other EvalNodes in the chain keep the kwargs used at the time
        of their creation.

        If the same instance of EvalNode occurs multiple times in the
        evaluation tree, the first occurence is copied and this copy is
        used on subsequent occurences. This should preserve loops in the
        tree. This is realized with the memo parameter which should be left
        default on manual calls to this method.

        It should be possible to copy multiple nodes having the same parents
        (resulting in to manual copy() calls) if one supplies the same dict
        to memo for both calls. (Start with an empty dict)
        """
        
        if last_parents is None:
            last_parents = {}

        if last_kwargs is None:
            last_kwargs = {}

        if memo is None:
            memo = {}

        if type(self.parents) is list:
            new_parents = [None] * len(self.parents)
        elif type(self.parents) is dict:
            new_parents = {}

        if self in memo:
            new = memo[self]
        else:
            for n, p in self.parents_iter:
                cpy_or_memo = p.copy(name_suffix, plot, ignore_cache, last_parents, last_kwargs, memo=memo)#, **kwargs)
                new_parents[n] = cpy_or_memo

            kwargs_use = copy.deepcopy(self.ext)

            if len(new_parents) == 0:
                new_parents = last_parents

                # merge items whose values are dicts
                for k, v in last_kwargs.items():
                    if k in kwargs_use and isinstance(v, MutableMapping):
                        kwargs_use[k] |= v
                    else:
                        kwargs_use[k] = v


            new = type(self)(self.name + name_suffix,
                    parents=new_parents,
                    plot=self.plot_on if plot is None else plot,
                    cache=self.cache,
                    ignore_cache=self.ignore_cache if ignore_cache is None else ignore_cache,
                    **kwargs_use
                )

            memo[self] = new

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

        Also notice that NodeGroup overrides this behaviour because
        the subscription of the do()-call return value is equivalent to
        subscripting the dictionary of parents of the NodeGroup.
        Since these two are generally not the same the default
        behaviour is as stated here and EvalNode.parents can be 
        subscripted for the other possible outcome.
        """
        return SubscriptedNode('_{}_subs_{}'.format(self.name, index), parents=self, subscript=index)

    def __contains__(self, item):
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
    def plot_on(self):
        """
        The plot property decides when a node is plotted. There
        are three possibilities:
         * True or False: The node is always or never plotted 
         * matplotlib.axes.Axes instance: The node is always plotted
           on the given Axes instance
         * object or [object]: Defines a "plot group", i.e. all
           nodes in the same group can be plotted at will. This means
           this node is plotted if the same object (or one object 
           from the list) to which plot is set is passed  to the 
           __call__() method of the node chain.

        See also: plot_on
        """
        if isinstance(self._plot, matplotlib.axes.Axes):
            return self._ax
        else:
            return self._plot

    @plot_on.setter
    def plot_on(self, on):
        if isinstance(on, matplotlib.axes.Axes):
            self._plot = True
            self._ax = on
        else:
            self._plot = on
            self._ax = None

    def in_plot_group(self, group):
        """
        Decides wether the node is plotted depending on the
        parameter passed to __call__.
        See the plot property for further information.
        """
        if self._plot is True or self._plot is False:
            return self._plot

        if isinstance(self._plot, list):
            if isinstance(group, list):
                return len([g for g in group if g in self._plot]) >= 1
            else:
                return group in self._plot
        else:
            if isinstance(group, list):
                return self._plot in group
            else:
                return self._plot == group or self._plot is group

    def __call__(self, ax=None, plot_on=None, memo=None, **kwargs):
        """
        Run the node and recursively its parents. 
         * ax: If not None eventually plot nodes on this Axes
               object.
         * plot_on: a plot group or a list of plot groups that 
               should be plotted. See plot_on property for more
               information. If one of None, True or False,
               exactly the nodes that are set to True are plotted.
         * memo: Used to prevent double plotting. Nodes that are
               in memo are ignored. Double evaluation is prevented
               by caching results in RAM.
         * kwargs: All nodes kwargs are joined with this kwargs
               dict for this call. Watch out because cached nodes
               are not re-called even if kwargs change.

        Returns: the memo list after running all parents.
        """
        common = {}

        memo = [] if memo is None else memo

        self._do(ax, plot_on, False, common, memo, kwargs)

        return memo

    def set(self, **kwargs):
        """
        Set kwargs of this instance.
        """
        self.ext = self.def_kwargs(self.ext | kwargs)

    def get_kwargs(self):
        """
        Get kwargs of this instance.
        To modify them it is better to use set()
        """
        return self.ext

    # get data, either from cache or by regenerating
    # if regenerated and a cache is attached to the 
    # eval node, the data is stored
    @simple_cache
    def _generate_v(self, common, kwargs):
        v = self.do(self._parent_data, common, **(self.ext | kwargs))
        if not self.cache is None:
            self.cache.store(self.name, v)

        return v

    @simple_cache
    def _cacheload_v(self):
        v = self.cache.load(self.name)
        return v

    def _update_common(self, common, kwargs):
        common_add = self.common(common, **(self.ext | kwargs))
        if not common_add is None:
            common.update(common_add)

    def _do(self, ax, plot_group, need_data, common, memo, kwargs):
        # check if a cache exists and if we should use it
        # or if we need data from our parents for regeneration
        if self.cache is None or self.ignore_cache or not self.cache.exists(self.name):
            this_need_data = True
        else:
            this_need_data = False

        self._update_common(common, kwargs)

        # request data from parents using this info
        must_fill = False # track if any parents return data (not None)
        for k, p in self.parents_iter:
            maybe_data = p._do(ax, plot_group, this_need_data, common, memo, kwargs)
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
                    self._parent_data[k] = p._do(None, plot_group, True, common, memo, kwargs)
            v = self._generate_v(common, kwargs)
            logging.debug("Evaluation chain: _do at {} chose to regenerate the data (1)".format(self.name))
        elif len(self.parents) == 0 and this_need_data:
            # if there are no parents but data is needed
            v = self._generate_v(common, kwargs)
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

        if self.in_plot_group(plot_group) and not self in memo:
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

            self.plot(v, plot_on, common, **(self.ext | kwargs))

        return v

    def search_parent(self, starts_with):
        """
        Search the chain of nodes for any nodes
        whose name starts with the given string
        """
        if str(self.name).startswith(starts_with):
            return self

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
        as kwargs but eventually overriden with the kwargs passed to
        the initial __call__() of the tree.
        """
        return kwargs

    def subclass_init(self, parents, **kwargs):
        """
        Override this method if you need control over how the 
        parents are attached this eval node or if you need
        to modify the parents in some way.

        You need to set self.parents manually in this method

        It is executed at instance __init__

        Using this method is discouraged because customizing
        parent attachment can mess with copy semantics.
        """
        # doing this is optional, if subclass_init returns
        # None, this is the default behaviour
        self.parents = parents

    def common(self, common, **kwargs):
        """
        Override this method to add information to the common dict
        that is shared across this instance and all parents and children
        in the evaluation chain. It is not intended to be used for data that is 
        expensive to calculate or store.

        You can either return a set of keys which are used to update the common
        dict or directly modify the parameter *common* given.

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

        kwargs are the kwargs that are set at creation time
        potentially overriden with those at call time.
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

    def __getitem__(self, key):
        return self.parents[key]

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

def copy_to_group(name, node, count=None, last_parents=None, last_kwargs=None, memo=None):
    """
    Create a NodeGroup instance grouping copies of the given node.
    If count is given, the node is copied *count* times with last_parents and last_kwargs.
    If count is not given at least one of last_parents and last_kwargs must be passed
    and must be either list or dict. If one of those is a dict the dict keys are used as 
    name suffixes for the copy call, if both are list the suffixes are enumerated.
    If both are dicts, the keys of last_parents are used.
    """
    if count is None:
        if last_parents is None and last_kwargs is None:
            raise ValueError("At least one of count, last_parents, last_kwargs")
        elif last_parents is None:
            last_parents = len(last_kwargs) * [None]
        elif last_kwargs is None:
            last_kwargs = len(last_parents) * [{}]
        elif len(last_parents) != len(last_kwargs):
            raise ValueError("When both last_parents and last_kwargs are given their length must be equal")

        if type(last_parents) is dict and (type(last_kwargs) is dict or type(last_kwargs) is list):
            iterator = [(key, par, kw) for (key, par), kw in zip(last_parents.items(), last_kwargs)]

        elif type(last_parents) is list and type(last_kwargs) is dict:
            iterator = [(key, par, kw) for (key, kw), par in zip(last_kwargs.items(), last_parents)]

        elif type(last_parents) is list and type(last_kwargs) is list:
            iterator = [(str(i), par, kw) for i, (par, kw) in enumerate(zip(last_parents, last_kwargs))]

        else:
            raise ValueError("last_parents and last_kwargs must be of type dict or list")
    else:
        iterator = [(str(i), last_parents, last_kwargs) for i in range(count)]

    group_members = {}
    for key, par, kw in iterator:
        cpy = node.copy(key, last_parents=par, memo=memo, **kw)
        group_members[key] = cpy
    
    return NodeGroup(name, parents=group_members)



class NodeSet(EvalNode):
    """
    OBSOLETE! Use copy_to_group for consistent node behaviour

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
        print("Warning: Using NodeSet is obsolete and may lead to inconsistent behaviour. See the docs for more information.")
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

