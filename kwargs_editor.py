import numpy as np
import logging
import time
import argparse
import inspect
import sys

from grapheval.cache import PickleNodeCache
from grapheval.graph import draw_node_chain


logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logging.getLogger('node.node').setLevel(logging.DEBUG)

# for old extractor based experiments
cache_opts = {
    'cachedir' : 'cache',
    'regenerate' : False
}

def drift():
    pass
def diffusion():
    pass
def split():
    pass
def boundaries():
    pass

if __name__ == "__main__":
    logging.info("Loading kwargs")
    if sys.argv[1] == '-p':
        import pickle
        with open(sys.argv[2], mode='rb') as f:
            print(pickle.load(f))
    else:
        cache = PickleNodeCache(cache_opts['cachedir'], sys.argv[1])
        oldkw = cache.load_kwargs(sys.argv[2]) 
        print("old kwargs: ", oldkw)

    if len(sys.argv) == 5:
        oldkw[sys.argv[3]] = eval(sys.argv[4])
        print("overwrite with those kwargs?", oldkw)
        if input("Y/N") == "Y":
            cache.store_kwargs(sys.argv[2], oldkw) 
