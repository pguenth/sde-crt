#ifndef BATCH_KRUELLS_H
#define BATCH_KRUELLS_H

#include <Eigen/Core>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <functional>
#include <limits>
#include "batch/wienerprocess.h"
#include "batch/scheme.h"
#include "batch/breakpoint.h"
#include "batch/integrator.h"
#include "batch/pseudoparticlebatch.h"

using namespace std::placeholders;

class BatchKruells1 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
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
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells8(std::map<std::string, double> params);
                ~BatchKruells8();
};

/**
 */
class BatchKruells9 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells9(std::map<std::string, double> params);
                ~BatchKruells9();
};

/**
 */
class BatchKruells10 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells10(std::map<std::string, double> params);
                ~BatchKruells10();
};

/**
 * BatchKruells9 with spatial limit L
 */
class BatchKruells11 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells11(std::map<std::string, double> params);
                ~BatchKruells11();
};

/**
 * BatchKruells9 with dkappa/dx from eq.(19)
 */
class BatchKruells12 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells12(std::map<std::string, double> params);
                ~BatchKruells12();
};

/**
 * BatchKruells9 with dkappa/dx from eq.(19) (other sign)
 */
class BatchKruells13 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells13(std::map<std::string, double> params);
                ~BatchKruells13();
};

/**
 * SemiImplicit
 */
class BatchKruells14 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                SemiImplicitWeakScheme2 *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells14(std::map<std::string, double> params);
                ~BatchKruells14();
};

/**
 * Implicit
 */
class BatchKruells15 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                ImplicitEulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells15(std::map<std::string, double> params);
                ~BatchKruells15();
};

/**
 * KPPC
 */
class BatchKruells16 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                KPPCScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruells16(std::map<std::string, double> params);
                ~BatchKruells16();
};

/**
 */
class BatchKruellsB1 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruellsB1(std::map<std::string, double> params);
                ~BatchKruellsB1();
};

/**
 */
class BatchKruellsB2 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                SemiImplicitWeakScheme2 *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruellsB2(std::map<std::string, double> params);
                ~BatchKruellsB2();
};
/**
 */
class BatchKruellsC1 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchKruellsC1(std::map<std::string, double> params);
                ~BatchKruellsC1();
};

/**
 */
class BatchAchterberg1 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg1(std::map<std::string, double> params);
                ~BatchAchterberg1();
};
/**
 */
class BatchAchterberg1KPPC : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                KPPCScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg1KPPC(std::map<std::string, double> params);
                ~BatchAchterberg1KPPC();
};

/**
 */
class BatchAchterberg2 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                EulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2(std::map<std::string, double> params);
                ~BatchAchterberg2();
};
/**
 */
class BatchAchterberg2KPPC : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                KPPCScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2KPPC(std::map<std::string, double> params);
                ~BatchAchterberg2KPPC();
};
/**
 */
class BatchAchterberg2Implicit : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                ImplicitEulerScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2Implicit(std::map<std::string, double> params);
                ~BatchAchterberg2Implicit();
};
/**
 */
class BatchAchterberg2SecondOrder : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                SecondOrderScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2SecondOrder(std::map<std::string, double> params);
                ~BatchAchterberg2SecondOrder();
};
/**
 */
class BatchAchterberg2SecondOrder2 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                SecondOrderScheme2 *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2SecondOrder2(std::map<std::string, double> params);
                ~BatchAchterberg2SecondOrder2();
};
/**
 */
class BatchAchterberg2SemiImplicit : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                SemiImplicitWeakScheme *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2SemiImplicit(std::map<std::string, double> params);
                ~BatchAchterberg2SemiImplicit();
};
/**
 */
class BatchAchterberg2SemiImplicit2 : public PseudoParticleBatch {
        private:
                BreakpointTimelimit *_tlimit;
                //BreakpointSpatial *_slimit;
                SemiImplicitWeakScheme2 *_scheme;
                WienerProcess *_process;
                //LinearIntegrator *_sintegrator;


        public:
                /**
                 * Required parameters:
                 */
                BatchAchterberg2SemiImplicit2(std::map<std::string, double> params);
                ~BatchAchterberg2SemiImplicit2();
};
#endif
