#include "stochasticprocess.h"

StochasticProcess::StochasticProcess(int ndim, void *seed) : _ndim(ndim) {}

//DummyProcess::DummyProcess(int ndim, int seed) : _ndim(ndim) {}
//Eigen::VectorXd DummyProcess::next(double timestep) {
//    Eigen::VectorXd(_ndim) v;
//	
//    int i = 0;
//    for (int i = 0; i < _ndim; i++){
//	v(i) = 0;
//    }
//
//    return v;
//}

