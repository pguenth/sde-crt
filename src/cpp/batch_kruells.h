#ifndef BATCH_KRUELLS_H
#define BATCH_KRUELLS_H

#include <Eigen/Core>
#include <vector>
#include <cmath>
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
                WienerProcess *_process;

        public:
                /**
                 * Required parameters:
                 */
                BatchKruells1(std::map<std::string, double> params);//, double x_min, double x_max);
                ~BatchKruells1();
};

/**
 */
class BatchKruells2 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;

        public:
                /**
                 * Required parameters:
                 */
                BatchKruells2(std::map<std::string, double> params);
                ~BatchKruells2();
};

/**
 */
class BatchKruells3 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells3(std::map<std::string, double> params);
                ~BatchKruells3();
};


/**
 */
class BatchKruells4 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells4(std::map<std::string, double> params);
                ~BatchKruells4();
};


/**
 */
class BatchKruells5 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells5(std::map<std::string, double> params);
                ~BatchKruells5();
};

/**
 */
class BatchKruells6 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                WienerProcess *_process;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells6(std::map<std::string, double> params);
                ~BatchKruells6();
};

/**
 */
class BatchKruells7 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells7(std::map<std::string, double> params);
                ~BatchKruells7();
};

/**
 */
class BatchKruells8 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells8(std::map<std::string, double> params);
                ~BatchKruells8();
};
#endif
