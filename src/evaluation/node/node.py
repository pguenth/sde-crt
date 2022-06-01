from abc import ABC, abstractmethod, ABCMeta
from collections.abc import MutableMapping
import copy
import logging
from functools import wraps
from itertools import count
import matplotlib.axes
from .cache import KwargsCacheMixin

logger = logging.getLogger(__name__)

def simple_cache(func):
    """
    Caches a function in RAM so that is only executed once
    for its lifetime. Calling it a second time with different
    arguments is NOT leading to executing the function a second
    time, instead the cache is returned, potentially generated with
    different arguments.

    If the object has an attribute called 'disable_ram_cache' and it
    is set to True at the time of the decorated function being called
    this decorator has no effect. I know this is not ideal but see it as
    a temporary solution (more in :py:class:`EvalNode`)

    If this functionality is not intended, use functools.cache
    """

    @wraps(func)
    def decorated(*args, **kwargs):
        self = args[0]
        if hasattr(self, 'disable_ram_cache') and self.disable_ram_cache:
            return func(*args, **kwargs)

        if not '__c_' + func.__name__ in self.__dict__:
            self._log_debug("[RAM cache] cache empty, calling the requested function {}".format(func.__name__))
            self.__dict__['__c_' + func.__name__] = func(*args, **kwargs)
        else:
            self._log_debug("[RAM cache] returning cached return value of the requested function {}".format(func.__name__))

        return self.__dict__['__c_' + func.__name__]

    return decorated

def find_line(tup):
    if isinstance(tup, matplotlib.lines.Line2D):
        return tup
    #!! can it be list or sth?
    elif isinstance(tup, tuple):
        for t in tup:
            fl = find_line(t)
            if not fl is None:
                return fl

        return None
    else:
        return None

def dict_or_list_iter(obj):
    if type(obj) is list:
        return enumerate(obj)
    elif type(obj) is dict:
        return obj.items()
    else:
        raise TypeError("obj must be either dict or list")

def dict_or_list_map(obj, func):
    if type(obj) is list:
        return [func(n) for n in obj]
    elif type(obj) is dict:
        return {k : func(n) for k, n in obj.items()}
    else:
        raise TypeError("obj must be either dict or list")

class EmptyParentKey:
    pass

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

    def __eq__(self, other):
        if not type(other) == type(self):
            return False

        return self.colors == other.colors and self._next == other._next

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

    At the moment modifying the nodes after calling one of them can lead to
    unexpected behaviour because the result of :py:meth:`do` is cached (in RAM)
    and eventual modifications cannot be detected. This may change in the
    future to support this workflow. Until then you can deactivate the RAM cache
    by setting :py:attr:`EvalNode.disable_ram_cache` to *True*.

    Every decendant class of this class should implement one (atomic) step
    or operation in the process of evaluating an experiment. When deriving
    an *EvalNode* the methods listed below can or should be overriden to provide
    the behaviour you intend.

    :param name: Name of this instance
    :type name: string

    :param parents: The parents of this *EvalNode*. Can also be understood as
        the data sources. *parent_data*
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

    :param cache_not_found_regenerate:  
        If no :py:class:`NodeCache` instance is attached and children did not
        request data, either
        
        * *True*: force a regeneration (which in turn forces all
          children to regenerate). Default.
        * *False*: do not generate anything and continue

    :type cache_not_found_regenerate: bool, optional

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

    def __init__(self, name, parents=None, plot=False, cache=None, ignore_cache=False, cache_not_found_regenerate=True, **kwargs):
        self._id = next(self._instance_ids)
        self._global_id = next(self._base_id)
        self.name = name
        self.ext = self.def_kwargs(**kwargs)
        self.cache = cache
        self.ignore_cache = ignore_cache

        self.cache_not_found_regenerate = cache_not_found_regenerate

        # does not propagate to children
        # set to true for inherited classes that are
        # only proving mappings etc. to prevent double caching
        #
        # !!! NOT USED ANYWHERE IN THIS CLASS
        # PLEASE IMPLEMENT IN _DO(...)
        self.never_cache = False

        # remember if the kwargs were identified as changed in one run of _do
        # so that other runs return the updated data instead of nothing
        self._kwargs_changed = False

        self.disable_ram_cache = False

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

        # access _parent_data keys while iterating by using _set_parent_data and
        # _get_parent_data to automatically handle the case of only a single parent
        if type(self.parents) is list:
            self._parent_data = [None] * len(self.parents)
        elif type(self.parents) is dict:
            self._parent_data = {}
        elif isinstance(self.parents, EvalNode) or self.parents is None:
            self._parent_data = None
        else:
            raise TypeError("Parents must be either list or dict or EvalNode or NoneType")

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
        elif isinstance(self.parents, EvalNode) or self.parents is None:
            new_parents = None
        else:
            raise TypeError("parents have an invalid type")

        if self in memo:
            new = memo[self]
        else:
            is_last_node = True
            for n, p in self.parents_iter:
                cpy_or_memo = p.copy(name_suffix, plot, ignore_cache, last_parents, last_kwargs, memo=memo)#, **kwargs)
                is_last_node = False
                if n is EmptyParentKey:
                    # this can only happen once because EmptyParentKey is used only if there's no more than one parents
                    new_parents = cpy_or_memo
                else:
                    new_parents[n] = cpy_or_memo

            kwargs_use = copy.deepcopy(self.ext)

            if is_last_node:
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
                    cache_not_found_regenerate=self.cache_not_found_regenerate, 
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
            for any type of parents (list or dict or :py:class:`EvalNode`).
            For lists, it is enumerate(:py:attr:`parents`), and for dict
            it is :py:attr:`parents`.items(). For a single parent of class
            :py:class:`EvalNode`, an iterator yielding
            :py:class:`EmptyParentKey`: *parent* is returned so that this
            property can be used uniformly as iterator.

        :rtype: iterator
        """
        if type(self.parents) is list:
            return enumerate(self.parents)
        elif type(self.parents) is dict:
            return self.parents.items()
        elif isinstance(self.parents, EvalNode):
            return {EmptyParentKey: self.parents}.items()
        elif self.parents is None:
            return {}.items()
        else:
            raise TypeError("parents must be either dict or list or EvalNode or None")

    def is_parent(self, possible_parent):
        """
        :return: Returns if a :py:class:`EvalNode` instance is a parent of this instance
        :rtype: bool
        """
        if type(self.parents) is list:
            return possible_parent in self.parents
        elif type(self.parents) is dict:
            return possible_parent in self.parents.values()
        elif isinstance(self.parents, EvalNode):
            return possible_parent is self.parents
        elif self.parents is None:
            return False
        else:
            raise TypeError("parents must be either dict or list or EvalNode instance or None")

    def parents_contains(self, key):
        """
        :return: True if the parents iterator contains an item with the given key.
            For parents stored as list, this is equal to 'key < len(parents)'
            and for parents stored as dict, this is eqal to 'key in parents'
            If the parents are a single :py:class:`EvalNode` this always returns
            False.

        :rtype: bool
        """
        if type(self.parents) is list:
            return key < len(self.parents)
        elif type(self.parents) is dict:
            return key in self.parents
        elif isinstance(self.parents, EvalNode):
            return False
        elif self.parents is None:
            return False
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
        if not (isinstance(parents, EvalNode) or type(parents) is list or 
                type(parents) is dict or parents is None):
            raise TypeError("invalid type for parents (got {})".format(type(parents)))

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
        if isinstance(self._do_plot, matplotlib.axes.Axes):
            return self._ax
        else:
            return self._do_plot

    @plot_on.setter
    def plot_on(self, on):
        if isinstance(on, matplotlib.axes.Axes):
            self._do_plot = True
            self._ax = on
        else:
            self._do_plot = on
            self._ax = None

    def in_plot_group(self, group):
        """
        :returns: Wether the node is plotted depending on the *plot_on*
            parameter passed to __call__. See :py:attr:`plot_on` for further
            information.
        :rtype: bool
        """
        if self._do_plot is True or self._do_plot is False:
            return self._do_plot

        if isinstance(self._do_plot, list):
            if isinstance(group, list):
                return len([g for g in group if g in self._do_plot]) >= 1
            else:
                return group in self._do_plot
        else:
            if isinstance(group, list):
                return self._do_plot in group
            else:
                return self._do_plot == group or self._do_plot is group

    @staticmethod
    def _default_common_dict():
        return {'_kwargs': {}, '_kwargs_by_type': {}, '_kwargs_by_instance': {}, '_colors': {}}

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
        common = self._default_common_dict()

        memo = [] if memo is None else memo

        self._plot(ax, plot_on, common, memo, kwargs)

        return memo

    def data_extra(self, common=None, **kwargs):
        """
        The same as :py:attr:`data` but allowing for kwargs and
        supplying a common dict (with gets filled eventually)

        :return: returned object from :py:meth:`do`
        """
        if common is None:
            common = self._default_common_dict()
        else:
            common |= {k : v for k, v in self._default_common_dict().items() if k not in common}

        return self._data(True, common, kwargs)

    @property
    def data(self):
        """
        Get the data of the node, eventually calling parent node's do method
        and using RAM cache

        :return: returned object from :py:meth:`do`
        """
        return self.data_extra()

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
        self._log_info("Executing .do() to regenerate data")
        v = self.do(self._parent_data, common, **(self.ext | kwargs))
        if not self.cache is None:
            self._log_debug("Caching generated data")
            self.cache.store(self.name, v)
            if hasattr(self.cache, 'store_kwargs'): # isinstance stopped working for nodes defined outside of the module, maybe because of some import stuff.
                self._log_debug("Caching kwargs")
                self.cache.store_kwargs(self.name, self.ext | kwargs)

        return v

    @simple_cache
    def _cacheload_v(self):
        self._log_info("Loading data from cache")
        v = self.cache.load(self.name)
        return v

    def _update_common(self, common, kwargs):
        common['_kwargs'].update({self.name: (self.ext | kwargs)})
        common['_kwargs_by_type'].update({type(self): (self.ext | kwargs)})
        common['_kwargs_by_instance'].update({self: (self.ext | kwargs)})

        common_add = self.common(common, **(self.ext | kwargs))
        if not common_add is None:
            common.update(common_add)

    def _check_data_needed(self):
        """
        Check if this node needs data from its parents
        checks existance and usability of caching
        """
        if self.cache is None:
            if self.cache_not_found_regenerate:
                self._log_debug("Needs data because no NodeCache is attached")
                return True
            else:
                self._log_debug("Does not need data")
                return False
        elif self.ignore_cache:
            self._log_debug("Needs data because cache should be ignored")
            return True
        elif not self.cache.exists(self.name) and self.cache_not_found_regenerate:
            self._log_debug("Needs data because no cached data exists")
            return True
        else:
            # cache exists, check kwargs
            if hasattr(self.cache, 'kwargs_changed') and self.cache.kwargs_changed(self.name, self.ext):
                self._log_info("The stored kwargs are not up to date, so the cache is ignored.")
                self._kwargs_changed = True
                return True
            else:
                self._log_debug("Does not need data")
                return False

    def _set_parent_data(self, key, data):
        """
        Sets the parent data of parent with key.
        Use this method to correctly handle the case of a single parent, i.e.
        the key being EmptyParentKey as returned by :py:attr:`parents_iter`.
        This enables concise iterating over the parents

        :return: *None*
        """

        if key is EmptyParentKey:
            self._parent_data = data
        else:
            self._parent_data[key] = data

    def _get_parent_data(self, key):
        if key is EmptyParentKey:
            return self._parent_data
        else:
            return self._parent_data[key]

    def _maybe_fill_parent_data(self, this_need_data, common, kwargs):
        """
        Fills parent data but allows parents to return None if this_need_data is False.
        This function is for checking wether any :py:class:`EvalNode`s see the need
        to update and if they do, update them.

        :return: Wether one or more parents actually returned data
        :rtype:
        """
        self._log_debug("Traversing the tree upwards")
        must_fill = False # track if any parents return data (not None)
        for k, p in self.parents_iter:
            maybe_data = p._data(this_need_data, common, kwargs)
            assert not (maybe_data is None and this_need_data) # if we requested data, but recieved None, quit
            self._set_parent_data(k, maybe_data)
            must_fill = must_fill or not maybe_data is None

        return must_fill

    def _fill_parent_data(self, common, kwargs):
        """
        Fills parent data (that is still empty) but 
         * no plotting
         * force parents to return data

        This function is used to gather the data of all remaining nodes
        if one or more nodes return an update while called in :py:meth:`_maybe_fill_parent_data`

        :rtype: None
        """
        self._log_debug("Some parents returned data, so we assume regeneration is neccessary and traverse the tree a second time to get all data")
        for k, p in self.parents_iter:
            if self._get_parent_data(k) is None:
                self._set_parent_data(k, p._data(True, common, kwargs))

    def _data(self, need_data, common, kwargs):
        # request data from parents using this info
        this_need_data = self._check_data_needed()
        self._update_common(common, kwargs)
        must_fill = self._maybe_fill_parent_data(this_need_data, common, kwargs)
        if must_fill:
            # if none or some (but not all) parents returned None (= (>=1 parent returned data))
            # we force those to return us data
            # if there is no cache, we will always land here
            # because we requested data from all parents (this_need_data == True in this case)
            self._fill_parent_data(common, kwargs)

            self._log_debug("Requesting regeneration or RAM cache retrieval of data")
            v = self._generate_v(common, kwargs)
        elif self.parent_count == 0 and this_need_data:
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
        elif self._kwargs_changed:
            self._log_debug("Request loading data from cache or RAM cache because the kwargs were marked as updated in another call of _do()")
            v = self._cacheload_v()
        else:
            # all parents returned None and the child doesn't need data
            # so we return None to tell it that we don't have updates
            v = None
            self._log_debug("No data retrieval neccessary from request or parents")

        return v

    def _plot(self, ax, plot_group, common, memo, kwargs):
        for k, p in self.parents_iter:
            self._log_debug("[_plot] traversing tree upwards")
            p._plot(ax, plot_group, common, memo, kwargs)

        if self.in_plot_group(plot_group) and not self in memo:
            memo.append(self)
            plot_on = self._ax if not self._ax is None else ax
            # modification: at this place, when v == None, v was loaded from 
            # and returned, so child notes regenerated.
            # this has been removed but I dont know if it was actually senseful
            self._log_debug("[_plot] Request loading data from cache or RAM cache anyways for plotting")
            v = self._data(True, common, kwargs)

            self._log_debug("[_plot] Plot the node")

            self._plot_handles = self.plot(v, plot_on, common, **(self.ext | kwargs))

            if not (type(self._plot_handles) is list or self._plot_handles is None):
                self._plot_handles = [self._plot_handles]

            # bodgy: store the color of the first returned handle
            if not self._plot_handles[0] is None and self._color is None:
                l = find_line(self._plot_handles[0])
                if not l is None:
                    self._color = l.get_color()
                else:
                    self._color = None

    def map_parents(self, callback, call_on_none=False):
        """
        Executes callback for every parent of this instance
        
        :param callback: The :py:class:`Callable` to be executed
        :type callback: :py:class:`Callable` (:py:class:`EvalNode`) -> object

        :param call_on_none: If this :py:class:`EvalNode` has no parents
            call the callback with *None* as argument.
        :type call_on_none: :py:class:`bool`, Default: *False*

        :return: object with structure like :py:attr:`parents` but the nodes
            replaced with the objects returned by *callback*
        :rtype: same as :py:attr:`parents`
        """
        if isinstance(self.parents, EvalNode):
            return callback(self.parents)
        elif self.parents is None:
            if call_on_none:
                return callback(None)
            else:
                return None
        else:
            return dict_or_list_map(self.parents, callback)

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
        self._search_tree_all(starts_with, l)
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

    def map_tree(self, map_callback, starts_with=None, return_values=None):
        """
        Execute *map_callback* for every node in this tree or for all nodes
        whose name starts with the given string, starting with self.
        
        :param map_callback: The callable to execute. The only argument passed is
            the :py:class:`EvalNode` instance in question.
        :type map_callback: :py:class:`Callable` (EvalNode) -> object

        :param starts_with: If None run the callback for every :py:class:`EvalNode`
            in the tree. If not None run callback only for :py:class:`EvalNode`
            instances whose name start with *starts_with*. Default: *None*
        :type starts_with: :py:class:`str` or *None*, optional

        :param return_values: The *dict* for collecting the return values in
            the recursive calls. Should be left at the default value for 
            normal usage.
        :type return_values: :py:class:`dict` or *None*, optional

        :return: the return values of every call.
        :rtype: dict {EvalNode -> object} 
        """
        if return_values is None:
            return_values = {}

        if starts_with is None or str(self.name).startswith(starts_with):
            return_values[self] = map_callback(self)

        for _, p in self.parents_iter:
            p.map_tree(map_callback, starts_with, return_values)

        return return_values

    @property
    def handles(self):
        return self._plot_handles

    @property
    def handles_complete_tree(self):
        m = self.map_tree(lambda n: n.handles)
        
        l = []
        for x in m.values():
            if not x is None:
                l += x

        return l

    @property
    def parent_count(self):
        if self.parents is None:
            return 0
        elif isinstance(self.parents, EvalNode):
            return 1
        else:
            return len(self.parents)

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

    def do(self, parent_data, common, **kwargs):
        """
        Override this method. Perform the required evaluation.
        parent_data is a dict containing the return values of 
        all parent's do calls with the keys as given at creation
        time.

        You can skip overriding this method. If not overridden the
        parent data is returned.

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
        return parent_data

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

        If no color is found, a new one is returned 
        """
        if not self._color is None:
            return self._color

        for _, p in self.parents_iter:
            p_color = p.get_color()
            if not p_color is None:
                return p_color

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
        #print(self.name, kwargs['subscript'])
        return parent_data[kwargs['subscript']]

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
        raise TypeError("Using NodePlotGroup is deprecated because it hasn't been updated, will probably not work and I don't know what its purpose was anymore.")
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

