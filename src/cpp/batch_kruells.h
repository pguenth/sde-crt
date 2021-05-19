#ifndef BATCH_KRUELLS_H
#define BATCH_KRUELLS_H

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <functional>
#include "batch/wienerprocess.h"
#include "batch/breakpoint.h"
#include "batch/integrator.h"
#include "batch/pseudoparticlebatch.h"

using namespace std::placeholders;

class BatchKruells1 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                BatchKruells1(double x0, double y0, int N, double Tmax, double Xsh, double a, double b);//, double x_min, double x_max);
                ~BatchKruells1();
};



#endif
