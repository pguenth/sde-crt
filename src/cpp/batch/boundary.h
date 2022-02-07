#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <Eigen/Core>
#include <stdexcept>
#include "pseudoparticlestate.h"


// a boundary implements the case of boundaries which are modifying the particle state
// for example, a mirroring boundary
class TrajectoryBoundary {
    protected:
        virtual void _interact(SpaceTimePoint& p) const = 0;
        virtual bool _check(const SpaceTimePoint& p) const = 0;

    public:
        void replace(SpaceTimePoint& particle_state) const; 
};

#endif
