import pickle
import logging
from abc import abstractmethod, ABC
from pathlib import Path

class CacheException(Exception):
    pass

class NodeCache:
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

class FileCache(NodeCache):
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

    def load(self, name):
        try:
            with open(self.cache_path(name), mode='rb' if self.byte_mode else 'r') as cachefile:
                logging.info("Using cached data from {} for {}".format(self.cache_path(name), name))
                content = type(self)._read_file(cachefile)
                logging.info("Cached data loaded")
        except IOError:
            raise CacheException("Cannot read the cache {} in {}".format(name, self.cache_path(name)))

        return content

    def store(self, name, data):
        with open(self.cache_path(name), mode='wb' if self.byte_mode else 'w') as cachefile:
            logging.info("Storing data in {} for {}".format(self.cache_path(name), name))
            type(self)._write_file(cachefile, data)
            logging.info("Generated data is stored")

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
