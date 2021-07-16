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


PseudoParticleState::PseudoParticleState() : _finished(false), _tracked(true), _breakpoint_state(BreakpointState::UNDEFINED) {}

PseudoParticleState::PseudoParticleState(const PseudoParticle *particle, const SpaceTimePoint& p0, int max_steps, bool tracked) :
        _particle(particle), _finished(false), _tracked(tracked), _breakpoint_state(BreakpointState::UNDEFINED)  {
    _trajectory.push_back(p0); // push directly into the trajectory to guarantee non-empty vector
    _pre_allocate(max_steps);
}

PseudoParticleState::PseudoParticleState(const PseudoParticle *particle, double t0, const Eigen::VectorXd& x0, int max_steps, bool tracked) :
        _particle(particle), _finished(false), _tracked(tracked), _breakpoint_state(BreakpointState::UNDEFINED)  {

    // push directly into the trajectory to guarantee non-empty vector
    SpaceTimePoint s(t0, x0);
    _trajectory.push_back(s);
    _pre_allocate(max_steps);
}

void PseudoParticleState::_pre_allocate(int max_steps){
    if (_tracked && max_steps > 0) {
        _trajectory.reserve(max_steps);
    }
}

// trajectory
const Eigen::VectorXd& PseudoParticleState::get_x() const {
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

bool PseudoParticleState::tracked() const {
    return _tracked;
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
PseudoParticleState::operator std::string() const{
    std::ostringstream s;
    s << (std::string)(_trajectory.back()) << "; Finished: " << finished() << "; BreakpointState: " << (int)_breakpoint_state;

    std::string str = s.str();
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    return str;
}

const SpaceTimePoint& PseudoParticleState::operator ()() const{
    return get_p();
}

void PseudoParticleState::update(const SpaceTimePoint& p){
    if (_finished){
        throw std::logic_error("Tried to modify a finished state");
    }

    if (_tracked){
        _trajectory.push_back(p);
    }else{
        _trajectory.back() = p;
    }
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

