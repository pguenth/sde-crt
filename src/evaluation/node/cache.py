import pickle
import logging
from abc import abstractmethod, ABC
from collections.abc import Sequence, Mapping, MutableSequence, MutableMapping
import numpy as np
from pathlib import Path
from astropy.units import Quantity, allclose
from enum import Enum, auto
import copy

class IsUnpicklable(Enum):
    GENERIC = auto()
    CALLABLE = auto()

class CacheException(Exception):
    pass

class NodeCache(ABC):
    """
    Handles caching of data calculated by nodes 
    (not yet: and tracks nodes that were already evaluated)
    """

    @abstractmethod
    def exists(self, name):
        """
        Return True if the is cached data existing under *name*,
        else return False.
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, name):
        """
        Load the data cache under the given *name*.
        Throw an CacheException if no cache called *name* 
        can be found.
        """
        raise NotImplementedError

    @abstractmethod
    def store(self, name, data):
        """
        Store the given *data* under *name*, potentially overriding existing data
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, name, callback, clear):
        """
        in-place caching: looks for an existing cache file under
        *name* and loads it. If it is not existing, *callback* is 
        executed, the return value is cached in this file and 
        returned.

        clear overrides this behaviour: if set to true, the 
        potentially existing cache file is ignored and overriden
        after regeneration via *callback*
        """
        raise NotImplementedError

class KwargsCacheMixin(ABC):
    @abstractmethod
    def _store_kwargs(self, name, kwargs):
        """
        Store the kwargs. Must be overridden.
        Don't call this method directly instead use :py:meth:`store_kwargs`
        """
        raise NotImplementedError

    @abstractmethod
    def _load_kwargs(self, name):
        """
        Load the kwargs. Must be overridden.
        """
        raise NotImplementedError

    def store_kwargs(self, name, kwargs):
        """
        Store the new kwargs (overriding the old)
        """
        kw = self._cleancopy(kwargs)
        self._store_kwargs(name, kw)

    def load_kwargs(self, name):
        """
        It is not advised to use this method, even if it is in priciple
        useful. You should only use :py:meth:`store_kwargs` and 
        :py:meth:`kwargs_changed`. Especially for comparing an possibly
        different kwargs use the latter method instead of this method
        and a manual comparison.
        """
        return self._load_kwargs(name)

    def kwargs_changed(self, name, kwargs):
        """
        Return True if the cached kwargs differ from the kwargs that have been stored

        Return False if they are the same.

        If no kwargs can be cached (NodeCache not supporting it) always return False
        """
        try:
            old_kw = self.load_kwargs(name)
        except NotImplementedError:
            return False
        except CacheException:
            logging.warning("No stored kwargs found. This may be the case because of migration from cached nodes that were cached without kwargs. This warning will be removed in the future and instead raise an error.")
            return False

        new_kw = self._cleancopy(kwargs)
        #logging.error("at node {}: old_kw: {} ****** new_kw: {}".format(name, old_kw, new_kw))
        if self._compare_dict_rec(old_kw, new_kw):
            return False
        else:
            return True

    def replace_kwargs_item(self, item):
        """
        If any possible content type of a kwargs dict (or sub-lists/dicts)
        is not cacheable by the class using this mixin, override this method
        so that those objects get overriden with some (arbitrary) placeholder.

        If not overridden nothing happens.
        """
        return item

    def _cleancopy(self, kwargs):
        kw = copy.deepcopy(kwargs)
        self._clean_kwargs(kw)
        return kw

    def _clean_kwargs(self, item):
        """
        Cleans a (possible nested dict/list structured) item by replacing all
        not-list/dict items using :py:meth:`_replace_kwargs_item`
        """
        if isinstance(item, MutableMapping):
            for k, v in item.items():
                x = self._clean_kwargs(v)
                if not x is None:
                    item[k] = x
        elif isinstance(item, str):
            return self.replace_kwargs_item(item)
        elif isinstance(item, MutableSequence):
            for i, v in enumerate(item):
                x = self._clean_kwargs(v)
                if not x is None:
                    item[i] = x
        else:
            return self.replace_kwargs_item(item)
        
    def _compare_dict_rec(self, old_kw, kwargs):
        if old_kw is None:
            if kwargs is {}:
                return True
            else:
                return False

        for k, v in kwargs.items():
            if not k in old_kw:
                logging.info("not in old kwargs: {}".format(k))
                return False
            elif not self._dynamic_value_compare(old_kw[k], v):
                logging.info("not the same value: key: {}; new value: {}; old value: {}".format(k, v, old_kw[k]))
                return False

        return True

    def _dynamic_value_compare(self, v1, v2):
        """
        This method compares two objects handling various types
            - np.ndarray
            - dict like
            - list like
            - everything else supporting __eq__

        """
        if not type(v1) is type(v2):
            return False
        if isinstance(v1, Quantity):
            return repr(v1.unit) == repr(v2.unit) and allclose(v1.value, v2.value)
        elif isinstance(v1, np.ndarray):
            return np.array_equal(v1, v2)
        elif isinstance(v1, Mapping):
            return self._compare_dict_rec(v1, v2)
        elif isinstance(v1, str):
            # str before Sequence because string is a sequence
            return v1 == v2
        elif isinstance(v1, Sequence):
            if not len(v1) == len(v2):
                return False
            for v1_, v2_ in zip(v1, v2):
                if not self._dynamic_value_compare(v1_, v2_):
                    return False
            return True
        else:
            return v1 == v2

class FileCache(KwargsCacheMixin, NodeCache):
    """
    Provides methods for caching and restoring with pickle.

    """
    def __init__(self, cache_dir='.', cache_name=''):
        self.path = "{}/{}".format(cache_dir, cache_name)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.extension = ""
        self.byte_mode = True

    def cache_path(self, name):
        return "{}/{}{}".format(self.path, name, self.extension)

    def exists(self, name):
        try:
            with open(self.cache_path(name), mode='rb' if self.byte_mode else 'r') as cachefile:
                return True
        except IOError:
            logging.info("Tried to access cache in {}, but nothing found.".format(self.cache_path(name)))
            return False

    @staticmethod
    def _read_file(file):
        raise NotImplementedError

    @staticmethod
    def _write_file(file, obj):
        raise NotImplementedError

    def _load_kwargs(self, name):
        return self.load(name + "_kwargs", loglevel=logging.DEBUG)

    def _store_kwargs(self, name, kwargs):
        self.store(name + "_kwargs", kwargs, loglevel=logging.DEBUG)

    def load(self, name, loglevel=logging.DEBUG):
        try:
            with open(self.cache_path(name), mode='rb' if self.byte_mode else 'r') as cachefile:
                logging.log(loglevel, "Using cached data from {} for {}".format(self.cache_path(name), name))
                content = type(self)._read_file(cachefile)
                logging.log(loglevel, "Cached data loaded")
        except IOError:
            raise CacheException("Cannot read the cache {} in {}".format(name, self.cache_path(name)))

        return content

    def store(self, name, data, loglevel=logging.DEBUG):
        with open(self.cache_path(name), mode='wb' if self.byte_mode else 'w') as cachefile:
            logging.log(loglevel, "Storing data in {} for {}".format(self.cache_path(name), name))
            type(self)._write_file(cachefile, data)
            logging.log(loglevel, "Generated data is stored")

    def get(self, name, callback, clear):
        cache_path = self.cache_path(name)

        # rest is from helpers.pickle_cache
        if not cache_path is None and clear is False:
            try:
                content = self.load(name)
            except CacheException:
                logging.info("Cachefile not existing.")
                content = None
        else:
            content = None

        if content is None:
            logging.info("No cached data found or regeneration requested. Generating data...")
            content = callback()
            try:
                self.store(name, content)
            except IOError:
                logging.error("Could not store cache")
            except TypeError:
                pass

        return content


class PickleNodeCache(FileCache):
    """
    Provides methods for caching and restoring with pickle.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extension = ".pickle"

    @staticmethod
    def _read_file(file):
        return pickle.load(file)

    @staticmethod
    def _write_file(file, obj):
        pickle.dump(obj, file)

    def replace_kwargs_item(self, item):
        try:
            pickle.dumps(item)
        except (pickle.PicklingError, AttributeError):
            return IsUnpicklable.GENERIC
        else:
            return item
