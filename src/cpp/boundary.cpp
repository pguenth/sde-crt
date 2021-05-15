#include "boundary.h"
#include "pseudoparticle.h"

void TrajectoryBoundary::replace(PseudoParticleState& particle_state) const {
    if (_check(particle_state)){
        _interact(particle_state);
    }
}
