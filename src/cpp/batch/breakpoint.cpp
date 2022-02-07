#include "breakpoint.h"

BreakpointTimelimit::BreakpointTimelimit(double T) : _T(T) {};

BreakpointState BreakpointTimelimit::check(const SpaceTimePoint& p) const {
    if (p.t >= _T){
        return BreakpointState::TIME;
    }else{
        return BreakpointState::NONE;
    }
}

int BreakpointTimelimit::estimate_max_steps(const double timestep, const SpaceTimePoint& start) const {
    return (_T - start.t) / timestep;
}

BreakpointSpatial::BreakpointSpatial(Eigen::VectorXd x_min, Eigen::VectorXd x_max){
    if (x_min.size() != x_max.size()){
        throw std::invalid_argument("x_min and x_max must be of the same dimensions");
    }

    for (int i = 0; i < x_min.size(); i++){
        _1d_breakpoints.push_back(BreakpointSpatialIndex(i, x_min(i), x_max(i)));
    }
}

BreakpointState BreakpointSpatial::check(const SpaceTimePoint& p) const {
    if (p.x.size() != _1d_breakpoints.size()){
        throw std::invalid_argument("particle_state.x and the specified boundaries must have the same dimensions");
    }

    BreakpointState l;
    for (auto& breakpoint : _1d_breakpoints){
        if ((l = breakpoint.check(p)) != BreakpointState::NONE){
            return l;
        }
    }

    return BreakpointState::NONE;
}

int BreakpointSpatial::estimate_max_steps(const double timestep, const SpaceTimePoint& start) const {

    int max_estimate = 1;
    double this_estimate;
    for (auto& breakpoint : _1d_breakpoints){
        if ((this_estimate = breakpoint.estimate_max_steps(timestep, start)) > max_estimate){
            max_estimate = this_estimate;
        }
    }

    return max_estimate;
}

BreakpointSpatialIndex::BreakpointSpatialIndex(int index, double x_min, double x_max) :
    _index(index), _x_min(x_min), _x_max(x_max) {}

BreakpointState BreakpointSpatialIndex::check(const SpaceTimePoint& p) const {
    if (p.x(_index) <= _x_min){
        return BreakpointState::LOWER;
    }else if (p.x(_index) >= _x_max){
        return BreakpointState::UPPER;
    }else{
        return BreakpointState::NONE;
    }
}

int BreakpointSpatialIndex::estimate_max_steps(const double timestep, const SpaceTimePoint& start) const {
    double dmax = abs(start.x(_index) - _x_max);
    double dmin = abs(start.x(_index) - _x_min);
    double dbigger = 0.0;

    if (dmax > dmin){
        dbigger = dmax;
    }else{
        dbigger = dmin;
    }
        
    return 1.2 * (dbigger / sqrt(timestep));
}
