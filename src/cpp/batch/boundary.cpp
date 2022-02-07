#include "boundary.h"
#include "pseudoparticle.h"

void TrajectoryBoundary::replace(SpaceTimePoint& p) const {
    if (_check(p)){
        _interact(p);
    }
}
