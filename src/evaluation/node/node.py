from abc import ABC, abstractmethod, ABCMeta
from collections.abc import MutableMapping
import copy
import logging
from functools import wraps
from itertools import count
import matplotlib.axes

logger = logging.getLogger(__name__)

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
            self._log_debug("[RAM cache] cache empty, calling the requested function {}".format(func.__name__))
            self.__dict__['__c_' + func.__name__] = func(*args, **kwargs)
        else:
            self._log_debug("[RAM cache] returning cached return value of the requested function {}".format(func.__name__))

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
    """
    This is the base class for every node in a node chain/tree.
    
    A node tree can be though of as a series of subsequent operations on some
    data. This data can come from running an experiment, reading a file,...
    and can be processed in an arbitrary number of small, well defined steps
    to yield results from this data. Every type of step is implemented as
    decendant of this class. Optionally, a step (=Node) can also show its 
    results by plotting them. Also *EvalNode* handles caching of every step
    if neccessary.

    The idea is to create a set of decendant classes of *EvalNode* for the steps
    you need for your evaluation. Then you create a tree of EvalNodes by
    creating instances and linking them in the order neccessary to achieve your
    desired result. This linking is achieved by the paremeter *parents* which
    can be understood as the source(s) of data needed for the respective
    EvalNode.

    After setting up the tree you can use any *EvalNode* as entry point for
    doing the operations defined by simply calling the instance. This leads to
    recursively acquiring data on all parent nodes and then evaluating the
    called node.

    Every decendant class of this class should implement one (atomic) step
    or operation in the process of evaluating an experiment. When deriving
    an *EvalNode* the methods listed below can or should be overriden to provide
    the behaviour you intend.

    :param name: Name of this instance
    :type name: string

    :param parents: The parents of this *EvalNode*. Can also be understood as
        the data sources. If a single *EvalNode* instance is passed, expect a
        list containing a single item for *parent_data*, else *parent_data*
        will have the same structure as this parameter with *EvalNodes*
        exchanged for the return value of :py:meth:`EvalNode.do`
    :type parents: list or dict or :py:class:`EvalNode`

    :param plot: Decide if the node should execute its :py:meth:`plot` method.
        If *True* or *False* :py:meth:`plot` is always/never called. If
        another object is given (preferably *string* but technically anything
        that can be compared) it defines a plot group. See :py:attr:`plot_on`
        for further reference.
    :type plot: bool or object or list

    :param cache: The :py:class:`NodeCache` instance that this *EvalNode*
        should use for caching. If *None* caching is disabled.
    :type cache: :py:class:`NodeCache`

    :param ignore_cache: Ignore possibly existing cached data and re-run
        :py:meth:`EvalNode.do` in any case. **Caution: this only prevents
        cache reading, not writing.**
    :type ignore_cache: bool

    :param cache_not_found_action:  What to do if no :py:class:`NodeCache` instance is attached and children did not request data
        
        * '*always_regen*': force a regeneration (which in turn forces all 
          children to regenerate)
        * '*ignore*': do not generate anything and continue.

    :type cache_not_found_action: string, optional

    :param \*\*kwargs: The kwargs are stored and are passed to the overridden
        functions and can be used for customisation of a derived class.

    **Derived classes can or should override the following abstract methods:**

     * :py:meth:`do` : Must be overriden. This is the entrypoint for performing the neccessary calculations etc.
     * :py:meth:`plot` : Optional. Show the data on a specific axis object.
     * :py:meth:`def_kwargs` : Optional. Set defaults for subclass-specific kwargs.
     * :py:meth:`common` : Optional. Add values to a dictionary shared across the chain.
     * :py:meth:`subclass_init` : Optional. Perform initialisation if neccessary. (Do not override :py:meth:`__init__`)
     * :py:meth:`__contains__` : Optional. As specified in the python object model docs. Return wether a key is contained in the return value of :py:meth:`do`.

    For further information on the methods to override refer to the respective
    documentation of the methods. Keep in mind that any one of these calls
    can recieve kwargs other than the ones passed on instance creation. These
    should be ignored.
    """

    _base_id = count(0)

    def __init__(self, name, parents=None, plot=False, cache=None, ignore_cache=False, cache_not_found_action='always_regen', **kwargs):
        self._id = next(self._instance_ids)
        self._global_id = next(self._base_id)
        self.name = name
        self.ext = self.def_kwargs(**kwargs)
        self.cache = cache
        self.ignore_cache = ignore_cache

        if cache_not_found_action == 'always_regen':
            self._no_cache_regen = True
        elif cache_not_found_action == 'ignore':
            self._no_cache_regen = False
        else:
            raise ValueError("cache_not_found_action must be either always_regen or ignore")

        # does not propagate to children
        # set to true for inherited classes that are
        # only proving mappings etc. to prevent double caching
        #
        # !!! NOT USED ANYWHERE IN THIS CLASS
        # PLEASE IMPLEMENT IN _DO(...)
        self.never_cache = False

        self._color = None

        self.plot_on = plot

        # store handles from plot
        self._plot_handles = None

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
        """
        An id unique to this instance only counting
        instances of the specific class
        """

        return self._id

    @property
    def global_id(self):
        """
        An id unique to this instance counting every instance of
        :py:class:`EvalNode` and its derived classes
        """

        return self._global_id

    def copy(self, name_suffix, plot=None, ignore_cache=None, last_parents=None, last_kwargs=None, memo=None):#, **kwargs):
        """
        Create a copy of this :py:class:`EvalNode` and all of its parents
        recursively.

        :param name_suffix: This suffix is appended to the :py:attr:`name` of
            this instance to create a new name for the copy.
        :type name_suffix: string

        :param plot: if not *None* the value of :py:attr:`plot_on` is overridden
            in the copy of this :py:class:`EvalNode` and all of its parents'
            copies. Default: *None*
        :type plot: bool or object or list, optional

        :param ignore_cache: if not *None* the value of :py:attr:`ignore_cache`
            is overridden in the copy of this :py:class:`EvalNode` and all of 
            its parents' copies. Default: *None*
        :type plot: bool, optional

        :param last_parents: The top-level :py:class:`EvalNode` instances are
            created with the value of this parameter as their parents. Default:
            *None*
        :type last_parents: list or dict or :py:class:`EvalNode`, optional

        :param last_kwargs: The top-level :py:class:`EvalNode` instances are
            updated with this parameter where values which are itself dicts
            are merged into the existing dicts. The \*\*kwargs of the 
            :py:class:`EvalNode` instance that is copied is deepcopied, which 
            may lead to unexpected behaviour. (Maybe add: optionally deepcopy 
            or copy). Other :py:class:`EvalNodes` in the chain keep the kwargs
            used at the time of their creation. Default: *None* 
        :type last_kwargs: dict, optional

        :param memo: If the same instance of EvalNode occurs multiple times in the
            evaluation tree, the first occurence is copied and this copy is
            used on subsequent occurences. This should preserve loops in the
            tree. This is realized with the memo parameter which should be left
            default on manual calls to this method.
        :type memo: dict, optional

        It should be possible to copy multiple nodes having the same parents
        (resulting in to manual copy() calls) if one supplies the same dict
        to memo for both calls. (Start with an empty dict)

        :returns: A copy of this instance
        :rtype: :py:class:`EvalNode`
        """
        if name_suffix[0] != '_':
            name_suffix = '_' + name_suffix
        
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
                    if k in kwargs_use and isinstance(v, MutableMapping) and isinstance(kwargs_use[k], MutableMapping):
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
        Returns a node which in turn 
        subscripts the return value of this nodes do() call 
        on evaluation. (and returns this subscripted value)

        This may be confusing but I found it to be very useful
        in constructing node chains. Think of it as using the *EvalNodes*
        as if they were their generated/returned values.

        This also implies that the object returned is kind of
        a "future", only containing actual subscripted data
        when the node chain is run. This also makes the functions
        :py:meth:`__contains__` and :py:meth:`__iter__` unavailable to the base
        class since it is not known to the base class what subclasses'
        :py:meth:`do` call might return.

        You are encouraged to override :py:meth:`__contains__` for custom 
        subclasses if senseful return values can be provided. Further information:

        * https://docs.python.org/3/reference/datamodel.html#emulating-container-types
        * https://docs.python.org/3/reference/expressions.html#membership-test-details

        An alternative way to use subscription of this class would be
        to return the subscripted :py:attr:`parents` value but since this
        is much simpler to realize manually than the behaviour outlined
        above I chose it to be the way it is.

        In fact, NodeGroup overrides this behaviour in exactly this way
        because both ways of subscription are equivalent here, i.e.
        the subscription of the :py:meth:`do`-call return value is 
        equivalent to subscripting the dictionary of parents of the NodeGroup.

        :rtype: :py:class:`SubscriptedNode`
        """
        return SubscriptedNode('_{}_subs_{}'.format(self.name, index), parents=self, subscript=index)

    def __contains__(self, item):
        """
        This function always raises an exception. For more
        information on that see :py:meth:`__getitem__`
        """
        raise TypeError("Generally, subscripting nodes works by subscripting the return value of do(). Therefore it cannot be known to the base class (EvalNode) if this value contains an item.")

    @property
    def parents_iter(self):
        """
        :return: An iterator that can be handled uniformly
            for any type of parents (list or dict).
            For lists, it is enumerate(:py:attr:`parents`), and for dict
            it is :py:attr:`parents`.items()

        :rtype: iterator
        """
        if type(self.parents) is list:
            return enumerate(self.parents)
        elif type(self.parents) is dict:
            return self.parents.items()
        else:
            raise TypeError("parents must be either dict or list")

    def is_parent(self, possible_parent):
        """
        :return: Returns if a :py:class:`EvalNode` instance is a parent of this instance
        :rtype: bool
        """
        if type(self.parents) is list:
            return possible_parent in self.parents
        elif type(self.parents) is dict:
            return possible_parent in self.parents.values()
        else:
            raise TypeError("parents must be either dict or list")

    def parents_contains(self, key):
        """
        :return: True if the parents iterator contains an item with the given key.
            For parents stored as list, this is equal to 'key < len(parents)'
            and for parents stored as dict, this is eqal to 'key in parents'

        :rtype: bool
        """
        if type(self.parents) is list:
            return key < len(self.parents)
        elif type(self.parents) is dict:
            return key in self.parents
        else:
            raise TypeError("parents must be either dict or list")

    @property
    def parents(self):
        """
        :return: The list or dict of parents of this node
        :rtype: list or dict
        """
        return self._parents

    @parents.setter
    def parents(self, parents):
        self._parents = parents

    @property
    def plot_on(self):
        """
        The plot property decides when a node is plotted. There
        are three possibilities:

         * *True* or *False*: The node is always or never plotted 
         * :py:class:`matplotlib.axes.Axes` instance: The node is always plotted
           on the given *Axes* instance
         * :py:class:`object` or list of :py:class`object` : Defines a "plot
           group", i.e. all nodes in the same group can be plotted in one call.
           This means this node is plotted if the same object (or one object 
           from the list) to which :py:attr:`plot_on` is set is passed to the 
           :py:meth:`__call__` method of initial :py:class:`EvalNode`.

        See also: :py:meth:`__init__` parameter *plot*
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
        :returns: Wether the node is plotted depending on the *plot_on*
            parameter passed to __call__. See :py:attr:`plot_on` for further
            information.
        :rtype: bool
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
        Run the node and recursively its parents and possibly plot the results.
        :param ax: If not *None* potentially plot nodes on this Axes object
        :type ax: *None* or :py:class:`matplotlib.axes.Axes`, optional
        :param plot_on: a plot group or a list of plot groups that 
            should be plotted. See :py:attr:`plot_on` for more
            information. If one of *None*, *True* or *False*,
            exactly the nodes for which :py:attr:`plot_on` are set to
            *True* are plotted.
        :type plot_on: see :py:attr:`plot_on`, optional
        :param memo: Used to prevent double plotting. Nodes that are
            in memo are ignored. Double evaluation is prevented
            by caching results in RAM.
        :type memo: list of :py:class:`EvalNode`
        :param \*\*kwargs: All nodes kwargs are joined with this kwargs
            dict for this call. Watch out because cached nodes
            are not re-called even if kwargs change. (At the moment)

        :return: the memo list after running all parents. This may change
            in the future. Options may be: the data returned by (1) the called node
            (2) all nodes
        :rtype: list
        """
        common = {'_kwargs': {}, '_kwargs_by_type': {}, '_kwargs_by_instance': {}, '_colors': {}}

        memo = [] if memo is None else memo

        self._do(ax, plot_on, False, common, memo, kwargs)

        return memo

    def set(self, **kwargs):
        """
        Set kwargs of this instance.
        """
        self.ext = self.def_kwargs(**(self.ext | kwargs))

    def get_kwargs(self):
        """
        Get kwargs of this instance.
        To modify them it is better to use set()
        """
        return self.ext
    
    def _log(self, level, message):
        logger.log(level, "[@node {}] {}".format(self.name, message))

    def _log_debug(self, message):
        self._log(logging.DEBUG, message)

    def _log_info(self, message):
        self._log(logging.INFO, message)

    def _log_warning(self, message):
        self._log(logging.WARNING, message)

    # get data, either from cache or by regenerating
    # if regenerated and a cache is attached to the 
    # eval node, the data is stored
    @simple_cache
    def _generate_v(self, common, kwargs):
        self._log_debug("Executing .do() to regenerate data")
        v = self.do(self._parent_data, common, **(self.ext | kwargs))
        if not self.cache is None:
            self._log_debug("Caching generated data")
            self.cache.store(self.name, v)

        return v

    @simple_cache
    def _cacheload_v(self):
        self._log_debug("Loading data from cache")
        v = self.cache.load(self.name)
        return v

    def _update_common(self, common, kwargs):
        common['_kwargs'].update({self.name: (self.ext | kwargs)})
        common['_kwargs_by_type'].update({type(self): (self.ext | kwargs)})
        common['_kwargs_by_instance'].update({self: (self.ext | kwargs)})

        common_add = self.common(common, **(self.ext | kwargs))
        if not common_add is None:
            common.update(common_add)

    def _do(self, ax, plot_group, need_data, common, memo, kwargs):
        # check if a cache exists and if we should use it
        # or if we need data from our parents for regeneration
        if self.cache is None:
            if self._no_cache_regen: 
                self._log_debug("Needs data because no NodeCache is attached")
                this_need_data = True
            else:
                self._log_debug("Does not need data")
                this_need_data = False
        elif self.ignore_cache:
            self._log_debug("Needs data because cache should be ignored")
            this_need_data = True
        elif not self.cache.exists(self.name) and self._no_cache_regen:
            self._log_debug("Needs data because no cached data exists")
            this_need_data = True
        else:
            self._log_debug("Does not need data")
            this_need_data = False

        self._update_common(common, kwargs)

        self._log_debug("Traversing the tree upwards")

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
            self._log_debug("Some parents returned data, so we assume regeneration is neccessary and traverse the tree a second time to get all data")
            for k, p in self.parents_iter:
                if self._parent_data[k] is None:
                    self._parent_data[k] = p._do(None, plot_group, True, common, memo, kwargs)

            self._log_debug("Requesting regeneration or RAM cache retrieval of data")
            v = self._generate_v(common, kwargs)
        elif len(self.parents) == 0 and this_need_data:
            # if there are no parents but data is needed
            self._log_debug("This node has no parents but it needs data")
            self._log_debug("Requesting regeneration or RAM cache retrieval of data")
            v = self._generate_v(common, kwargs)
        elif need_data:
            # all parents returned None, this means there are no updates
            # in all our ancestors. Since the child calling us needs data
            # we can load the cache
            self._log_debug("Request loading data from cache or RAM cache")
            v = self._cacheload_v()
        else:
            # all parents returned None and the child doesn't need data
            # so we return None to tell it that we don't have updates
            v = None
            self._log_debug("No data retrieval neccessary from request or parents")

        if self.in_plot_group(plot_group) and not self in memo:
            memo.append(self)
            plot_on = self._ax if not self._ax is None else ax
            if v is None:
                # if there is need for a plot, but no data available at this point
                # we can use the cache for plotting since there are no updates
                self._log_debug("Request loading data from cache or RAM cache anyways for plotting")
                v = self._cacheload_v()
            else:
                # plot with available data
                self._log_debug("Plot the node")

            self._plot_handles = self.plot(v, plot_on, common, **(self.ext | kwargs))

            if not (type(self._plot_handles) is list or self._plot_handles is None):
                self._plot_handles = [self._plot_handles]

        return v

    def search_parent(self, starts_with):
        """
        Search the parents recursively for any nodes
        whose name starts with the given string
        and return the first match found
        """
        for _, p in self.parents_iter:
            found_parent = p.search_tree(starts_with) 
            if not found_parent is None:
                return found_parent

        return None

    def search_tree(self, starts_with):
        """
        Search the chain of nodes for any nodes
        whose name starts with the given string
        and return the first match found
        """
        if str(self.name).startswith(starts_with):
            return self

        for _, p in self.parents_iter:
            found_parent = p.search_tree(starts_with) 
            if not found_parent is None:
                return found_parent

        return None

    def _search_tree_all(self, starts_with, l):
        if str(self.name).startswith(starts_with):
            if not self in l:
                l.append(self)

        for _, p in self.parents_iter:
            p._search_tree_all(starts_with, l) 

    def search_tree_all(self, starts_with):
        """
        Search the chain for any nodes whose name starts with the given
        string and return all matches.
        """
        l = []
        self._search_parent_all(starts_with, l)
        return l

    def search_parents_all(self, starts_with):
        """
        Search the parents recursively for any nodes whose name starts with the given
        string and return all matches.
        """
        l = []
        for _, p in self.parents_iter:
            l += p.search_tree_all(starts_with)
        return l

    def is_descendant_of(self, possible_parent):
        """
        Return True if this instance has *possible_parent* anywhere in their
        parent chains
        """
        if self.is_parent(possible_parent):
            return True

        for _, p in self.parents_iter:
            if p.is_descendant_of(possible_parent):
                return True

    def map_tree(self, map_callback, return_values=None):
        """
        Execute *map_callback* for every node in this tree.
        Starting with self. Return a dict {EvalNode -> object}
        containing the return values of every call.
        """
        if return_values is None:
            return_values = {}

        return_values[self] = map_callback(self)
        for _, p in self.parents_iter:
            p.map_tree(map_callback, return_values)

        return return_values

    @property
    def handles(self):
        return self._plot_handles

    @property
    def handles_complete_tree(self):
        m = self.map_tree(lambda n: n.handles)
        print(m)
        
        l = []
        for x in m.values():
            if not x is None:
                l += x

        return l


    def def_kwargs(self, **kwargs):
        """
        Override this method if you have default values for optional
        kwargs or other things you want to do to the kwargs.
        Other overriden methods will see the dict returned by this method
        as kwargs but eventually overriden with the kwargs passed to
        the initial __call__() of the tree.

        Always be aware that unexpected kwargs may be passed to any
        function, so always include **kwargs in the call signature 
        even if your node does not use any kwargs or you write them
        out.
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

        Always be aware that unexpected kwargs may be passed to any
        function, so always include **kwargs in the call signature 
        even if your node does not use any kwargs or you write them
        out.
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

        Always be aware that unexpected kwargs may be passed to any
        function, so always include * **kwargs * in the call signature 
        even if your node does not use any kwargs or you write them
        out.
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

        Always be aware that unexpected kwargs may be passed to any
        function, so always include **kwargs in the call signature 
        even if your node does not use any kwargs or you write them
        out.
        """
        raise NotImplementedError

    def plot(self, data, ax, common, **kwargs):
        """
        Override this method if this eval node can plot something.
        It is called if requested with the return value from do(..)
        and an axes object to plot on.

        Should return a list of handles that this Node created.

        Always be aware that unexpected kwargs may be passed to any
        function, so always include **kwargs in the call signature 
        even if your node does not use any kwargs or you write them
        out.
        """
        raise NotImplementedError

    #def get_color(self, common):
    #    """
    #    Bodgy way to synchronize colors. Use only in conjunction with set_color.

    #    common['_colors'] stores which node used which color and if a node
    #    wants to sync its color, get_color() searches for nodes in this store
    #    which are parents of the node and returns the color of this node.
    #    """
    #    for node, color in common['_colors'].items():
    #        if self.is_descendant_of(node):
    #            return color
        
    #    return None
    #def set_color(self, color, common):
    #    """
    #    See get_color()
    #    """
    #    common['_colors'][self] = color

    def get_color(self):
        """
        New bodgy way to synchronise colors. Use only in conjunction with
        set_color.

        Colors are stored in a instance variable per node and the color of the
        first parent found that has this instance variable set is returned.

        If no color is found, None is returned
        """
        if not self._color is None:
            return self._color

        for _, p in self.parents_iter:
            return p.get_color()

        return None

    def set_color(self, color):
        """
        See get_color()
        """
        self._color = color

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

class LambdaNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        cb = kwargs['callback']

        if isinstance(parent_data, list) and len(parent_data) == 1:
            return cb(parent_data[0])
        elif isinstance(parent_data, list):
            return_list = []
            for v in parent_data:
                return_list.append(v)
            return return_list
        elif isinstance(parent_data, dict) and len(parent_data == 1):
            return cb(list(parent_data.values())[0])
        elif isinstance(parent_data, dict):
            return_dict = {}
            for k, v in parent_data.items():
                return_dict[k] = cb(v)
            return return_dict

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

