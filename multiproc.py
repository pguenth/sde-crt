from multiprocessing import Process

# should approximately be used as follows

#nsplit = 10
#splitted_points = chains.get_splitted_points(points, nsplit)
#
#if __name__ == '__main__' and len(sys.argv) > 1:
#    multiproc.run_multiproc(splitted_points)
#
#mergedpoints = PointsMergeNode(points.name + "_merged", splitted_points, cache=cache, ignore_cache=False)

# if you give some (random) cmd line argument (or replace the condition with something
# more useful) the splits will individually be simulated and the program will
# quit after running all of them

# after that restart without cmd line argument and you will see the behaviour as if
# no splitting has occured

# of course use mergedpoints instead of points in the following nodes
# automatic regeneration of the splitted nodes is still working as usual
# but with no performance improvements through multiprocessing

def run_multiproc(splitted_points, exit_on_finish=True):
    def get_splitted(splitted_p):
        splitted_p.data

    # if a number is given only run this split part and quit
    ps = []
    for sp in splitted_points:
        p = Process(target=get_splitted, args=(sp,))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    if exit_on_finish:
        exit()

def run_argv(splitted_points):
    splitted_points[int(sys.argv[1])].data
