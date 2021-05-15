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

typedef std::function<Eigen::VectorXd(Eigen::VectorXd&)> drift_t;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd&)> diffusion_t;
//typedef Eigen::VectorXd (*drift_t)(const Eigen::VectorXd&);
//typedef Eigen::MatrixXd (*diffusion_t)(const Eigen::VectorXd&);


typedef struct PseudoParticleOptions {
    double timestep = 1;

    std::list<TrajectoryBreakpoint *> breakpoints;
    std::list<TrajectoryBoundary *> boundaries;
    std::list<TrajectoryLiveIntegrator *> integrators;

    StochasticProcess *process = nullptr;
} PseudoParticleOptions;

typedef struct PseudoParticleCallbacks {
    drift_t drift;
    diffusion_t diffusion;

    PseudoParticleCallbacks(drift_t drift, diffusion_t diffusion) :
        drift(drift), diffusion(diffusion) {}
    PseudoParticleCallbacks() {}
} PseudoParticleCallbacks;

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
