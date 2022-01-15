import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
import proplot as pplt
import logging
from powerlawstudy import PowerlawSeries

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pls = PowerlawSeries('boundary_11', PyBatchKruells11, 'L', [3, 5, 6, 8, 10, 20, 30, 50, 80, 200, 1000], 
            {
              'r' : 4,
              't_inj' : 0.0005,
              'x0' : 0,
              'y0' : 1,
              'k_syn' : 0,
              'Tmax' : 300,
              'dt':  0.05,
              'Xsh' : 0.25,
              'beta_s' : 0.57,
              'q' : 5
            },
            confine_x = 2,
            reeval = False,
            param_human_name = "L",

        )
pls.get_series()
pls.get_histograms()
