#include "pseudoparticle.h"

void PseudoParticle::_construct(PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options){
    _callbacks = callbacks;
    _options = options;

    if (_options.process == nullptr) {
        _options.process = new WienerProcess(start.x.size(), 0);
    }

    int max_steps = estimate_max_steps(start);
    _state = PseudoParticleState(this, start, max_steps);
}

PseudoParticle::PseudoParticle(PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options){
   _construct(callbacks, start, options);
}

PseudoParticle::PseudoParticle(PseudoParticleCallbacks callbacks, double t0, Eigen::VectorXd x0, PseudoParticleOptions options){
    SpaceTimePoint start{t0, x0};
    _construct(callbacks, start, options);
}


PseudoParticle::PseudoParticle(drift_t drift, diffusion_t diffusion, SpaceTimePoint start, PseudoParticleOptions options){
    PseudoParticleCallbacks callbacks{drift, diffusion};
    _construct(callbacks, start, options);
}

PseudoParticle::PseudoParticle(drift_t drift, diffusion_t diffusion, double t0, Eigen::VectorXd x0, PseudoParticleOptions options){
    PseudoParticleCallbacks callbacks{drift, diffusion};
    SpaceTimePoint start{t0, x0};
    _construct(callbacks, start, options);
}


//get
const PseudoParticleState& PseudoParticle::state() const {
    return _state;
}

bool PseudoParticle::finished() const {
    return _state.finished();
}

const std::list<TrajectoryLiveIntegrator *> PseudoParticle::integrators() const {
    return _options.integrators;
}

//functions
bool PseudoParticle::_break(){
    BreakpointState b_state;
    for (auto& b : _options.breakpoints){
        if ((b_state = b->check(state())) != BreakpointState::NONE){
            _state.finish(b_state);
            return true;
        }
    }

    return false;
}

void PseudoParticle::_integrate(){
    for (auto& i : _options.integrators){
        i->integrate(state(), _options.timestep);
    }
}

//run
const PseudoParticleState& PseudoParticle::run() {
    while (!finished()){
        step();
    }

    return state();
}

bool PseudoParticle::step() {
    if (finished()){
       throw std::logic_error(".step() is not allowed on finished particles");
    }

    _integrate();
    if (_break()) return finished();

    Eigen::VectorXd x_old = state().get_x();
    Eigen::VectorXd x_new = x_old + _options.timestep * _callbacks.drift(x_old) + _callbacks.diffusion(x_old) * _options.process->next(_options.timestep);

    for (auto& b : _options.boundaries){
       b->replace(_state);
    }

    _state.update(state().get_t() + _options.timestep, x_new);

    return finished();
}
const std::list<TrajectoryBreakpoint *> PseudoParticle::breakpoints() const{
    return _options.breakpoints;
}


int PseudoParticle::estimate_max_steps(const SpaceTimePoint& start) const{
    int max_steps = 1;
    int this_steps;

    for (auto& breakpoint : _options.breakpoints){
        this_steps = breakpoint->estimate_max_steps(_options.timestep, start);
        if (this_steps > max_steps) max_steps = this_steps;
    }

    return max_steps;
}
