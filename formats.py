import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.nodefigure import NodeFigureFormat
import proplot as pplt

momentumhist = NodeFigureFormat(
                subplots=None,
                fig_format={'xscale': 'log', 'yscale': 'log', 'ylabel':'particle number density', 'yformatter': pplt.SciFormatter(), 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'},
                legend_kw={'loc': 'r', 'ncols': 1}
        )

doublehist = NodeFigureFormat(
                subplots={'ncols': 2},
                fig_format={'yscale': 'log', 'ylabel':'particle number density', 'yformatter': pplt.SciFormatter()},
                axs_format=[
                        {'xlabel': '$x$'},
                        {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

doublehistSED = NodeFigureFormat(
                subplots={'ncols' : 3},
                fig_format={},
                axs_format=[
                        {'yformatter': 'sci', 'xlabel': '$x$', 'yscale': 'log', 'ylabel':'particle number density'},
                        {'yformatter': 'sci', 'xscale': 'log', 'xformatter': 'sci', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'particle number density'},
                        {'yformatter': 'sci', 'xlabel': '$\\nu$', 'xformatter': 'sci', 'xscale': 'log', 'ylabel': 'Flux $\\nu F_\\nu$', 'yscale': 'log'}
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

histsandpowerlaw = NodeFigureFormat(
                subplots={'ncols': 3},
                axs_format=[
                        {'xlabel': '$x$', 'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                        {'xscale': 'log', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

histsandpowerlaw2 = NodeFigureFormat(
                subplots={'array' : [[1, 3, 5], [2, 4, 5]]},
                axs_format=[
                        {'xlabel': '$x$', 'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                        {'xlabel': '$x$', 'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                        {'xscale': 'log', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                        {'xscale': 'log', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )
