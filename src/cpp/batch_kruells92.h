#ifndef BATCH_KRUELLS92_H
#define BATCH_KRUELLS92_H

#include <Eigen/Core>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <functional>
#include "batch/wienerprocess.h"
#include "batch/breakpoint.h"
#include "batch/integrator.h"
#include "batch/pseudoparticlebatch.h"

using namespace std::placeholders;

class BatchKruells921 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                BatchKruells921(double x0, double y0, int N, double Tmax, double Tesc);//, double x_min, double x_max);
                ~BatchKruells921();
};

// Reproducing Fig. 2 of the 92 paper
class BatchKruells922 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                BatchKruells922(double x0, double y0, int N, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s);
                ~BatchKruells922();
};

// Same as 922 but with continous injection of PSEUDOparticles
class BatchKruells923 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                BatchKruells923(double x0, double y0, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s);
                ~BatchKruells923();
};

// Same as 922 but with continous injection of particles using an amplitude and an injection region
class BatchKruells924 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                //BatchKruells924(double x0, double p0, int N, double dx_inj, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s);
                BatchKruells924(std::map<std::string, double> params);
                ~BatchKruells924();
};
#endif
