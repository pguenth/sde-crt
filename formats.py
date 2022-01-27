import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.nodefigure import NodeFigureFormat
import proplot as pplt

doublehist = NodeFigureFormat(
                subplots={'ncols': 2},
                fig_format={'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                axs_format=[
                        {'xlabel': '$x$'},
                        {'xscale': 'log', 'xlabel': '$p/p_\\textrm{inj}$'}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

powerlaws = NodeFigureFormat(
                subplots=None,
                fig_format={'ylabel':'Powerlaw index $s$'},
                axs_format=[
                        {'xlabel': ''},
                ],
                legend_kw={}
        )

