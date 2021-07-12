from .extractor import *

class TrajectoryExtractor(Extractor):
    def __init__(self, index, trajectory_number, **kwargs):
        self.options = {
        }
        self.options.update(kwargs)
        super().__init__(**self.options)

        self.index = index
        self.trajectory_number = trajectory_number

    def data(self, experiment):
        t = experiment.states[self.trajectory_number].trajectory
        return np.array([(p.t, p.x[self.index]) for p in t]).T

    def plot(self, experiment, ax):
        t, x = self.data(experiment)
        ax.plot(t, x)
        
class MultiTrajectoryExtractor(Extractor):
    def __init__(self, index, count, **kwargs):
        self.options = {
        }
        self.options.update(kwargs)
        super().__init__(**self.options)

        self.index = index
        self.count = count

    def data(self, experiment):
        trajectories = []
        for s in experiment.states:
            if len(trajectories) >= self.count:
                break

            trajectories.append(np.array([(p.t, p.x) for p in s.trajectory]).T)

        return trajectories

    def plot(self, experiment, ax):
        trajectories = self.data(experiment)
        for t, x in trajectories:
            ax.plot(t, x)
