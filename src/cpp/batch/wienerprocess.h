#ifndef WIENERPROCESS_H
#define WIENERPROCESS_H

#include <Eigen/Core>
#include <random>
#include <math.h>
#include "pcg/pcg_random.hpp"
#include "stochasticprocess.h"


class WienerProcess: public StochasticProcess {
    protected:
        pcg32 _rng;
        std::normal_distribution<double> _dist;

    public:
        WienerProcess(int ndim, void *seed);
        Eigen::VectorXd next(double timestep);
};

#endif
