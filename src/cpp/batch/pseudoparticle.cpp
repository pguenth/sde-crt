#include "pseudoparticle.h"

PseudoParticleOptions::PseudoParticleOptions(){}

PseudoParticleOptions::~PseudoParticleOptions(){
    for (auto &elem : _integrators) delete elem;
    _integrators.clear();
}


PseudoParticleOptions::PseudoParticleOptions (const PseudoParticleOptions& p){
    tracked = p.tracked;
    breakpoints = p.breakpoints;
    scheme = p.scheme;
    seeds = p.seeds;

    _integrators = std::list<TrajectoryLiveIntegrator *>();

    for (auto liveint : p.integrators()){
        _integrators.push_back(liveint->clone());
    }

}

PseudoParticleOptions& PseudoParticleOptions::operator= (const PseudoParticleOptions& p){
    tracked = p.tracked;
    breakpoints = p.breakpoints;
    scheme = p.scheme;
    seeds = p.seeds;

    for (auto &elem : _integrators) delete elem;
    _integrators.clear();

    for (auto &liveint : p.integrators()){
        _integrators.push_back(liveint->clone());
    }

    return *this;
}

PseudoParticleOptions::PseudoParticleOptions (PseudoParticleOptions&& p){
    tracked = p.tracked;
    breakpoints = p.breakpoints;
    scheme = p.scheme;
    seeds = p.seeds;

    _integrators= std::list<TrajectoryLiveIntegrator *>();
    for (auto liveint : p.integrators()){
        _integrators.push_back(liveint->clone());
    }
}

PseudoParticleOptions& PseudoParticleOptions::operator= (PseudoParticleOptions&& p){
    breakpoints = p.breakpoints;
    scheme = p.scheme;
    seeds = p.seeds;
    
    for (auto &elem : _integrators) delete elem;
    _integrators.clear();

    _integrators = p.integrators();

    return *this;
}

const std::list<TrajectoryLiveIntegrator *>& PseudoParticleOptions::integrators() const {
    return _integrators;
}

void PseudoParticleOptions::add_integrator(const TrajectoryLiveIntegrator& integrator){
    _integrators.push_back(integrator.clone());
}

void PseudoParticle::_construct(SpaceTimePoint start, PseudoParticleOptions options){
    _options = options;

    if (_options.scheme == nullptr) {
        throw std::logic_error("No scheme process is given");
    }

    _options.scheme = _options.scheme->copy(_options.seeds);

    int max_steps;
    if (_options.tracked){
        max_steps = estimate_max_steps(start);
    }else{
        max_steps = 1;
    }
    _state = PseudoParticleState(this, start, max_steps, _options.tracked);
}

PseudoParticle::PseudoParticle(SpaceTimePoint start, PseudoParticleOptions options){
   _construct(start, options);
}

PseudoParticle::PseudoParticle(double t0, Eigen::VectorXd x0, PseudoParticleOptions options){
    SpaceTimePoint start{t0, x0};
    _construct(start, options);
}


//PseudoParticle::PseudoParticle(drift_t drift, diffusion_t diffusion, SpaceTimePoint start, PseudoParticleOptions options){
//    PseudoParticleCallbacks callbacks{drift, diffusion};
//    _construct(start, options);
//}
//
//PseudoParticle::PseudoParticle(drift_t drift, diffusion_t diffusion, double t0, Eigen::VectorXd x0, PseudoParticleOptions options){
//    PseudoParticleCallbacks callbacks{drift, diffusion};
//    SpaceTimePoint start{t0, x0};
//    _construct(start, options);
//}


//get
const PseudoParticleState& PseudoParticle::state() const {
    return _state;
}

const SpaceTimePoint& PseudoParticle::get_p() const {
    return _state.get_p();
}

bool PseudoParticle::finished() const {
    return _state.finished();
}

const std::list<TrajectoryLiveIntegrator *> PseudoParticle::integrators() const {
    return _options.integrators();
}

//functions
bool PseudoParticle::_break(){
    BreakpointState b_state;
    for (auto& b : _options.breakpoints){
        if ((b_state = b->check(get_p())) != BreakpointState::NONE){
            _state.finish(b_state);
            return true;
        }
    }

    return false;
}

void PseudoParticle::_integrate(){
    for (auto& i : _options.integrators()){
        i->integrate(get_p(), _options.scheme->timestep_at(state().get_p()));
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

    SpaceTimePoint new_p = _options.scheme->propagate(state().get_p());

    for (auto& b : _options.boundaries){
       b->replace(new_p);
    }

    _state.update(new_p);

    return finished();
}
const std::list<TrajectoryBreakpoint *> PseudoParticle::breakpoints() const{
    return _options.breakpoints;
}


int PseudoParticle::estimate_max_steps(const SpaceTimePoint& start) const{
    int max_steps = 1;
    int this_steps;

    for (auto& breakpoint : _options.breakpoints){
        this_steps = breakpoint->estimate_max_steps(_options.scheme->timestep_at(start), start);
        if (this_steps > max_steps) max_steps = this_steps;
    }

    return max_steps;
}
