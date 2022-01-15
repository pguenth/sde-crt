#include "wienerprocess.h"

WienerProcess::WienerProcess(int ndim, std::vector<uint64_t> seeds) : StochasticProcess(ndim) {
    if (seeds.size() < ndim){
        _unseeded = true;
    }else{
        _unseeded = false;
        for (int i = 0; i < ndim; i ++){
            _rngs.push_back(pcg32_unique(seeds.at(i)));
        }
    }
    _dist = std::normal_distribution<double>(0.0);
}

Eigen::VectorXd WienerProcess::next(double timestep){
    if (_unseeded){
        throw std::logic_error("next() cannot be called on an unseeded process. Make a copy of the process providing enough seeds.");
    }

    Eigen::VectorXd v(_ndim);

    for (int i = 0; i < _ndim; i++){
        v(i) = _dist(_rngs.at(i)) * sqrt(timestep);
    }

    return v;
}

StochasticProcess *WienerProcess::copy(std::vector<uint64_t> seeds){
    return new WienerProcess(_ndim, seeds);
}
