#ifndef BREAKPOINT_H
#define BREAKPOINT_H

#include <Eigen/Core>
#include <list>
#include "pseudoparticlestate.h"
#include "breakpointstate.h"

class PseudoParticleState;

// trajectory breakpoints implement conditions which lead to
// the stop of the particle simulation, for example the time reaching its limit
// or the spatial boundary is reached in escape-boundary scenarios
// for mirroring boundaries etc. see TrajectoryBoundary
class TrajectoryBreakpoint {
    public:
        virtual BreakpointState check(const SpaceTimePoint& p) const = 0;

        // estimates the maximum number of steps the pseudo particle might need
        // to reach this breakpoint (if the breakpoint is the only thing in the system)
        // used for pre-allocating the std::vector that stores the particle trajectory
        virtual int estimate_max_steps(const double timestep, const SpaceTimePoint& start) const = 0;
};

// breakpoint for time limited simulation
class BreakpointTimelimit : public TrajectoryBreakpoint { 
    private:
        double _T;

    public:
        BreakpointTimelimit(double T);
        BreakpointState check(const SpaceTimePoint& p) const;
        int estimate_max_steps(const double timestep, const SpaceTimePoint& start) const;
};

// looks at one specified dimension and breaks if limits exceed
class BreakpointSpatialIndex : public TrajectoryBreakpoint {
    private:
        int _index;
        double _x_min;
        double _x_max;

    public:
        BreakpointSpatialIndex(int index, double x_min, double x_max);
        BreakpointState check(const SpaceTimePoint& p) const;
        int estimate_max_steps(const double timestep, const SpaceTimePoint& start) const;
};

// breaks if at least one dimension exceeds the one of the limits
// use this class for 1D problems and ND problems
class BreakpointSpatial : public TrajectoryBreakpoint {
    private:
        std::list<BreakpointSpatialIndex> _1d_breakpoints;

    public:
        BreakpointSpatial(Eigen::VectorXd x_min, Eigen::VectorXd x_max);
        BreakpointState check(const SpaceTimePoint& p) const;
        int estimate_max_steps(const double timestep, const SpaceTimePoint& start) const;
};
    


#endif
