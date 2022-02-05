#ifndef PSEUDOPARTICLE_H
#define PSEUDOPARTICLE_H

#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "stochasticprocess.h"
#include "pseudoparticlestate.h"
#include "wienerprocess.h"
#include "integrator.h"
#include "boundary.h"
#include "breakpoint.h"

class TrajectoryBreakpoint;
class TrajectoryBoundary;
class TrajectoryIntegrator;

// maybe make those to accept SpaceTimePoint instead of vector to enable time-dep behaviour
typedef std::function<Eigen::VectorXd(Eigen::VectorXd&)> drift_t;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd&)> diffusion_t;
//typedef Eigen::VectorXd (*drift_t)(const Eigen::VectorXd&);
//typedef Eigen::MatrixXd (*diffusion_t)(const Eigen::VectorXd&);


typedef struct PseudoParticleOptions {
    private:
        std::list<TrajectoryLiveIntegrator *> _integrators; //owned by this class

    public:
        double timestep = 1;
        bool tracked = true;

        std::list<TrajectoryBreakpoint *> breakpoints;
        std::list<TrajectoryBoundary *> boundaries;

        StochasticProcess *process = nullptr;
        std::vector<uint64_t> seeds;

        PseudoParticleOptions();
        ~PseudoParticleOptions();
        PseudoParticleOptions (const PseudoParticleOptions& p);
        PseudoParticleOptions& operator= (const PseudoParticleOptions& p);
        PseudoParticleOptions (PseudoParticleOptions&& p);
        PseudoParticleOptions& operator= (PseudoParticleOptions&& p);

        const std::list<TrajectoryLiveIntegrator *>& integrators() const;
        void add_integrator(const TrajectoryLiveIntegrator& integrator);
} PseudoParticleOptions;

typedef struct PseudoParticleCallbacks {
    // maybe add timestep callback here for variable time-steps
    drift_t drift;
    diffusion_t diffusion;

    PseudoParticleCallbacks(drift_t drift, diffusion_t diffusion) :
        drift(drift), diffusion(diffusion) {}
    PseudoParticleCallbacks() {}
} PseudoParticleCallbacks;

/**
 * Representation of one pseudo particle 
 */
class PseudoParticle {
    private:
        //properties
        PseudoParticleCallbacks _callbacks;
        PseudoParticleOptions _options;

        PseudoParticleState _state;

        //functions
        bool _break();
        void _integrate();
        
        void _construct(PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options);

    public:
        //constructors
        /**
         * Constructs the pseudo particle from its callbacks, its starting point and an options struct
         */
        PseudoParticle(PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options);
        PseudoParticle(PseudoParticleCallbacks callbacks, double t0, Eigen::VectorXd x0, PseudoParticleOptions options);
        PseudoParticle(drift_t drift, diffusion_t diffusion, SpaceTimePoint start, PseudoParticleOptions options);
        PseudoParticle(drift_t drift, diffusion_t diffusion, double t0, Eigen::VectorXd x0, PseudoParticleOptions options);
        //PseudoParticle();
        //~PseudoParticle(); //destructor should be unneccessary

        //get
        const PseudoParticleState& state() const;
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
