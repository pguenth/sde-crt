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
                WienerProcess *_process;

        public:
                /**
                 * Required parameters:
                 * double x0, double y0, int N, double Tmax, double Tesc
                 */
                BatchKruells921(std::map<std::string, double> params);//, double x_min, double x_max);
                ~BatchKruells921();
};

/**
 * Reproducing Fig. 2 of the 92 paper
 */
class BatchKruells922 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;

        public:
                /**
                 * Required parameters:
                 * double x0, double y0, int N, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s
                 */
                BatchKruells922(std::map<std::string, double> params);
                ~BatchKruells922();
};

/**
 * Same as BatchKruells922 but with continous injection of PSEUDOparticles
 */
class BatchKruells923 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 * double x0, double y0, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s
                 */
                BatchKruells923(std::map<std::string, double> params);
                ~BatchKruells923();
};

/**
 * Same as BatchKruells922 but with continous injection of particles using an amplitude and an injection region
 */
class BatchKruells924 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;


        public:
                /**
                 * Required parameters:
                 * double x0, double p0, int N, double dx_inj, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s
                 */
                BatchKruells924(std::map<std::string, double> params);
                ~BatchKruells924();
};


/**
 * Same as BatchKruells923 but with kappa_bar (resp. Kparallel in 92 paper)
 * dependent on x or beta respectively
 */
class BatchKruells925 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 * double x0, double y0, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s
                 */
                BatchKruells925(std::map<std::string, double> params);
                ~BatchKruells925();
};
#endif
