#ifndef BATCH_H
#define BATCH_H

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "wienerprocess.h"
#include "breakpoint.h"
#include "integrator.h"
#include "pseudoparticlebatch.h"

class BatchSourcetest : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                BreakpointSpatial *_slimit;
                WienerProcess *_process;
                LinearIntegrator *_sintegrator;


        public:
                BatchSourcetest(double x0, int N, double Tmax, double x_min, double x_max);
                ~BatchSourcetest();
                std::vector<double> integrate();
};



#endif
