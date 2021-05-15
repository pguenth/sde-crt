#include "pseudoparticlestate.h"

SpaceTimePoint::SpaceTimePoint(double t, const Eigen::VectorXd& x) :
    t(t), x(x) {}

SpaceTimePoint::SpaceTimePoint() {}

SpaceTimePoint::operator std::string() const {
    std::ostringstream s;
    s << "(t|x) = (" << t << "|" << x << ")";
    return s.str();
}

//PseudoParticleState::PseudoParticleState(const PseudoParticle& particle) : _particle(particle) { }

PseudoParticleState::PseudoParticleState() : _finished(false) {}

PseudoParticleState::PseudoParticleState(const PseudoParticle *particle, const SpaceTimePoint& p0, int max_steps) :
        _particle(particle), _finished(false) {
    update(p0);
    _pre_allocate(max_steps);
}

PseudoParticleState::PseudoParticleState(const PseudoParticle *particle, double t0, const Eigen::VectorXd& x0, int max_steps) :
        _particle(particle), _finished(false) {

    update(t0, x0);
    _pre_allocate(max_steps);
}

void PseudoParticleState::_pre_allocate(int max_steps){
    if (max_steps > 0) {
        _trajectory.reserve(max_steps);
    }
}

// trajectory
const Eigen::VectorXd& PseudoParticleState::get_x() const {
    //std::cout << "x: " << get_p().x << "\n";
    return get_p().x;
}

double PseudoParticleState::get_t() const {
    return get_p().t;
}

const SpaceTimePoint& PseudoParticleState::get_p() const {
    if (_trajectory.size() == 0) {
        throw std::logic_error("Accessed an uninitialized state");
    }

    return _trajectory.back();
}

const trajectory_t& PseudoParticleState::get_trajectory() const {
    // not really nice to return a reference and not a copy
    // consider returning a copy instead
    return _trajectory;
}

// end state
bool PseudoParticleState::finished() const {
    return _finished;
}

const PseudoParticle* PseudoParticleState::get_particle() const {
    return _particle;
}

BreakpointState PseudoParticleState::get_breakpoint_state() const {
    return _breakpoint_state;
}

// operators
PseudoParticleState::operator std::string() {
    std::ostringstream s;
    s << (std::string)_trajectory.back() << "; Finished: " << _finished << "; BreakpointState: " << (int)_breakpoint_state;
    return s.str();
}

const SpaceTimePoint& PseudoParticleState::operator ()() const{
    return get_p();
}

void PseudoParticleState::update(const SpaceTimePoint& p){
    if (_finished){
        throw std::logic_error("Tried to modify a finished state");
    }

    _trajectory.push_back(p);
}

void PseudoParticleState::update(double t, const Eigen::VectorXd& x){
    SpaceTimePoint s(t, x);
    update(s);
}

void PseudoParticleState::finish(BreakpointState breakpoint_state) {
    if (_finished){
        throw std::logic_error("Tried to modify a finished state");
    }

    _breakpoint_state = breakpoint_state;
    _finished = true;
}

