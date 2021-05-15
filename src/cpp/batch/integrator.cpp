#include "integrator.h"

LinearLiveIntegrator::LinearLiveIntegrator(std::function<double(const Eigen::VectorXd&)> callback) :
     _value(0), _callback(callback) {}

void LinearLiveIntegrator::integrate(const PseudoParticleState& particle_state, double timestep){
    _value += _callback(particle_state.get_x()) * timestep;
}

double LinearLiveIntegrator::value() const{
    return _value;
}

LinearIntegrator::LinearIntegrator(std::function<double(const Eigen::VectorXd&)> callback) :
    _callback(callback) {}

double LinearIntegrator::integrate(trajectory_t trajectory) const{
    double value = 0;
    for (int i = 1; i < trajectory.size(); i++){
        value += _callback(trajectory[i].x) * (trajectory[i].t - trajectory[i - 1].t);
    }

    return value;
}


