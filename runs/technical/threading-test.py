from grapheval.node import *
import time
from grapheval.nodefigure import *
from formats import *
import logging

logging.basicConfig(level=logging.DEBUG, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logging.getLogger("matplotlib").setLevel(level=logging.INFO)

class ExceptionNode(EvalNode):
    def plot(self, data, ax, common, **kwargs):
        print("plot")
        return ["asdf"]

    def do(self, parent_data, common, **kwargs):
        for i in range(10):
            time.sleep(0.5)
            print("ex", self.name, i)
        print("do")
        return 1
        
class DummyNode(EvalNode):
    def plot(self, data, ax, common, **kwargs):
        print("dummy plot")
        return ["asdf2"]

    def do(self, parent_data, common, **kwargs):
        for i in range(10):
            time.sleep(0.5)
            print("dummy", self.name, i)
        print("do")
        print("dummy do")
        return 0

dnodes = []
for i in range(10):
    e = ExceptionNode(f"e{i}", plot="plt", threaded=True)
    d = DummyNode(f"d{i}", parents=e, plot="plt", threaded=True)
    dnodes.append(d)

p = DummyNode("p", parents=dnodes, plot="plt", threaded=True)

nfig = NodeFigure(momentumhist)
nfig.add(p, 0, plot_on="plt")
nfig.savefig("testthread.pdf")
