#ifndef BATCH_SOURCETEST_H
#define BATCH_SOURCETEST_H

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "batch/wienerprocess.h"
#include "batch/breakpoint.h"
#include "batch/integrator.h"
#include "batch/pseudoparticlebatch.h"

class BatchSourcetest : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                BreakpointSpatial *_slimit;
                WienerProcess *_process;
                LinearIntegrator *_sintegrator;


        public:
                BatchSourcetest(std::map<std::string, double> params);
                ~BatchSourcetest();
                std::vector<double> integrate();
};



#endif
