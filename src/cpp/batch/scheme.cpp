#include "scheme.h"

SDEScheme::SDEScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    _drift(drift), _diffusion(diffusion), _timestep(timestep), _process(process) {}

double SDEScheme::timestep_at(const SpaceTimePoint& p) const {
    return _timestep(p);
}


Eigen::VectorXd SDEScheme::next_random(const SpaceTimePoint& p) const {
    return _process->next(_timestep(p));
}

Eigen::VectorXd SDEScheme::next_random(double timestep) const {
    return _process->next(timestep);
}

EulerScheme::EulerScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDEScheme(drift, diffusion, timestep, process) {}

SpaceTimePoint EulerScheme::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    Eigen::VectorXd x_new = p.x + ts * _drift(p) + _diffusion(p) * next_random(ts);   
    return SpaceTimePoint(p.t + ts, x_new);
}

Scheme *EulerScheme::copy(std::vector<uint64_t> seeds) const {
    StochasticProcess *new_p = _process->copy(seeds);
    return new EulerScheme(_drift, _diffusion, _timestep, new_p);
}
