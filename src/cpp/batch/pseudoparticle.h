#ifndef PSEUDOPARTICLE_H
#define PSEUDOPARTICLE_H

#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "scheme.h"
#include "pseudoparticlestate.h"
#include "wienerprocess.h"
#include "integrator.h"
#include "boundary.h"
#include "breakpoint.h"

class TrajectoryBreakpoint;
class TrajectoryBoundary;
class TrajectoryIntegrator;

// maybe make those to accept SpaceTimePoint instead of vector to enable time-dep behaviour
//typedef Eigen::VectorXd (*drift_t)(const Eigen::VectorXd&);
//typedef Eigen::MatrixXd (*diffusion_t)(const Eigen::VectorXd&);


typedef struct PseudoParticleOptions {
    private:
        std::list<TrajectoryLiveIntegrator *> _integrators; //owned by this class

    public:
        bool tracked = true;

        std::list<TrajectoryBreakpoint *> breakpoints;
        std::list<TrajectoryBoundary *> boundaries;
        std::vector<uint64_t> seeds;

        Scheme *scheme = nullptr;

        PseudoParticleOptions();
        ~PseudoParticleOptions();
        PseudoParticleOptions (const PseudoParticleOptions& p);
        PseudoParticleOptions& operator= (const PseudoParticleOptions& p);
        PseudoParticleOptions (PseudoParticleOptions&& p);
        PseudoParticleOptions& operator= (PseudoParticleOptions&& p);

        const std::list<TrajectoryLiveIntegrator *>& integrators() const;
        void add_integrator(const TrajectoryLiveIntegrator& integrator);
} PseudoParticleOptions;

/**
 * Representation of one pseudo particle 
 */
class PseudoParticle {
    private:
        //properties
        PseudoParticleOptions _options;

        PseudoParticleState _state;

        //functions
        bool _break();
        void _integrate();
        
        void _construct(SpaceTimePoint start, PseudoParticleOptions options);

    public:
        //constructors
        /**
         * Constructs the pseudo particle from its callbacks, its starting point and an options struct
         */
        PseudoParticle(SpaceTimePoint start, PseudoParticleOptions options);
        PseudoParticle(double t0, Eigen::VectorXd x0, PseudoParticleOptions options);
        //PseudoParticle(SpaceTimePoint start, PseudoParticleOptions options);
        //PseudoParticle(double t0, Eigen::VectorXd x0, PseudoParticleOptions options);
        //PseudoParticle();
        //~PseudoParticle(); //destructor should be unneccessary

        //get
        const PseudoParticleState& state() const;
        const SpaceTimePoint& get_p() const;
        bool finished() const;
        const std::list<TrajectoryLiveIntegrator *> integrators() const;
        const std::list<TrajectoryBreakpoint *> breakpoints() const;

        // returns an estimate of the maximum number of steps this particle will run 
        // until reaching a breakpoint when running from start
        int estimate_max_steps(const SpaceTimePoint& start) const;

        //run
        const PseudoParticleState& run();
        bool step();
};

#endif
