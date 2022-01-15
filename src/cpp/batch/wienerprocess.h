#ifndef WIENERPROCESS_H
#define WIENERPROCESS_H

#include <Eigen/Core>
#include <random>
#include <map>
#include <vector>
#include <math.h>
#include "pcg/pcg_random.hpp"
#include "stochasticprocess.h"


class WienerProcess: public StochasticProcess {
    protected:
        std::vector<pcg32_unique> _rngs;
        std::normal_distribution<double> _dist;
        bool _unseeded;

    public:
        WienerProcess(int ndim, std::vector<uint64_t> seeds = {});
        Eigen::VectorXd next(double timestep);
        StochasticProcess *copy(std::vector<uint64_t> seeds);
};

#endif
