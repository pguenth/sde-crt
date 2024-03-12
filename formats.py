from grapheval.nodefigure import NodeFigureFormat
import proplot as pplt

pplt.rc['font.family'] = 'serif'
#pplt.rc['suptitle.pad'] = 18.0

momentumhist = NodeFigureFormat(
                subplots=None,
                fig_format={'xscale': 'log', 'yscale': 'log', 'ylabel':'Particle number density', 'yformatter': pplt.SciFormatter(), 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'},
                legend_kw={'loc': 'r', 'ncols': 1}
        )

doublehist = NodeFigureFormat(
                subplots={'ncols': 2},
                fig_format={'yscale': 'log', 'ylabel':'Particle number density', 'yformatter': 'log'}, #yformatter was pplt.SciFormatter() but produced 1x10^2 instead of just 10^2
                axs_format=[
                        {'xlabel': '$x$'},
                        {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

doublehist2 = NodeFigureFormat(
                subplots={'ncols': 2},
                fig_format={'yscale': 'log', 'ylabel':'Particle number density', 'yformatter': 'log'}, #yformatter was pplt.SciFormatter() but produced 1x10^2 instead of just 10^2
                axs_format=[
                        {'xlabel': '$x$'},
                        {'xscale': 'log', 'xformatter': 'log', 'xlabel': '$p/p_\\textrm{inj}$'}
                ]
        )

doublehistand2d = NodeFigureFormat(
                subplots={'ncols': 3},
                fig_format={'yscale': 'log', 'yformatter': 'log'},
                axs_format=[
                        {'xlabel': '$x$', 'ylabel':'Particle number density'},
                        {'xscale': 'log', 'xformatter': 'log', 'ylabel':'Particle number density', 'xlabel': '$p/p_\\textrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$y$'}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

doublehistand2d2 = NodeFigureFormat(
                subplots={'array': [[1, 2], [3, 3], [3, 3]]},
                fig_format={'yscale': 'log', 'yformatter': 'log'},
                axs_format=[
                        {'xlabel': '$x$ (Shock at $x=0$)', 'ylabel':'Particle number density'},
                        {'xscale': 'log', 'xformatter': 'log', 'xlabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'}
                ],
        )

doublehistand2dmany = NodeFigureFormat(
                subplots={'array': [[1, 2], [3, 4], [5, 6]]},
                fig_format={'yscale': 'log', 'yformatter': 'log'},
                axs_format=[
                        {'xlabel': '$x$ (Shock at $x=0$)', 'ylabel':'Particle number density'},
                        {'xscale': 'log', 'xformatter': 'log', 'xlabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'}
                ],
        )

contours4 = NodeFigureFormat(
                subplots={'array': [[1, 2], [3, 4]]},
                fig_format={'yscale': 'log', 'yformatter': 'log'},
                axs_format=[
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'},
                        {'xlabel' : '$x$', 'ylabel': '$p/p_\\mathrm{inj}$'}
                ],
        )

doublehistSED = NodeFigureFormat(
                subplots={'ncols' : 3},
                fig_format={},
                axs_format=[
                        {'yformatter': 'sci', 'xlabel': '$x$', 'yscale': 'log', 'ylabel':'Particle number density'},
                        {'yformatter': 'sci', 'xscale': 'log', 'xformatter': 'sci', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'Particle number density'},
                        {'yformatter': 'sci', 'xlabel': '$\\nu$ in Hz', 'xformatter': 'sci', 'xscale': 'log', 'ylabel': 'Flux $\\nu F_\\nu$ (a.u.)', 'yscale': 'log'}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

singlehistSED = NodeFigureFormat(
                subplots={'ncols' : 2},
                fig_format={},
                axs_format=[
                        {'yformatter': 'log', 'xscale': 'log', 'xformatter': 'log', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'Particle number density'},
                        {'yformatter': 'log', 'xlabel': '$\\nu$ in Hz', 'xformatter': 'log', 'xscale': 'log', 'ylabel': 'Flux $\\nu F_\\nu$ (a.u.)', 'yscale': 'log'}
                ],
                #legends_kw={0: {'loc': 'ur', 'ncols': 1}}
        )


powerlaws = NodeFigureFormat(
                subplots=None,
                fig_format={'ylabel':'Power-law index $s$'},
                axs_format=[
                        {'xlabel': ''},
                ],
                legend_kw={}
        )

histsandpowerlaw = NodeFigureFormat(
                subplots={'ncols': 3},
                axs_format=[
                        {'xlabel': '$x$', 'yscale': 'log', 'ylabel':'Particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                        {'xscale': 'log', 'xlabel': '$p/p_\\textrm{inj}$', 'yscale': 'log', 'ylabel':'Particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()}
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

triplehist = NodeFigureFormat(
        subplots={'ncols' :3, 'nrows': 1, 'sharex': False},
        fig_format={'yscale': 'log', 'yformatter': 'log'},
        axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$x$ (parallel to shock)'},
                    {'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$z$ (perpendicular to shock)'},
                    {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}],
        legends_kw=[None, None, {'loc': 'ur', 'ncols': 1}]
    )
