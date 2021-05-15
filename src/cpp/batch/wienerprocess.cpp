#include "wienerprocess.h"

WienerProcess::WienerProcess(int ndim, void *seed) : StochasticProcess(ndim) {
    _rng = pcg32(*(static_cast<pcg32::state_type *>(seed)));
    _dist = std::normal_distribution<double>(0.0);
}

Eigen::VectorXd WienerProcess::next(double timestep){
    Eigen::VectorXd v(_ndim);

    for (int i = 0; i < _ndim; i++){
        v(i) = _dist(_rng) * sqrt(timestep);
    }

    return v;
}
