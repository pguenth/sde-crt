import sys
sys.path.insert(0, 'lib')
from pybatch.special.kruells import *
import proplot as pplt
import logging
import chains
import formats
from grapheval.cache import PickleNodeCache

from powerlawseries import *

from dtstudy import *
from tmaxstudy import *

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

name = "9a"
batch_cls = PyBatchKruells9

# use the results from the other studies
pls_dt = get_dtstudy(name, batch_cls)
pls_tmax = get_tmaxstudy(name, batch_cls)

pls_dt.label_template = "Varying $\\Delta\\tau$, $\\Delta x_\\mathrm{{adv}}=\\Delta\\tau/2$"
pls_tmax.label_template = "Varying $T$, $\\Delta x_\\mathrm{{adv}}=1/2$"

#creating one figure

nfig = NodeFigure(formats.powerlaws, xlabel='Runtime $T$', xscale='log', 
                xformatter=pplt.SimpleFormatter(precision=3, prefix="$", suffix="$"))
nfig.add(pls_tmax.datarow_chain)
nfig.format(suptitle='Reaching the temporal equilibrium')
ox = nfig[0].altx()
ox.invert_xaxis()
ox.format(xscale='log', xlabel="Timestep $\\Delta\\tau$")
pls_dt.datarow_chain(ox, plot_kwargs={'color': 'red'})
nfig[0].legend(handles=pls_dt.datarow_chain._plot_handles + pls_tmax.datarow_chain._plot_handles, ncols=1)
nfig.pad(.2)
nfig[0].annotate('$\\delta =0.5,~~\\sigma=0.5$', (0.61, 0.3), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
nfig._legends_kw = {}
nfig.savefig("figures/equilibrium_{}.pdf".format(name))
