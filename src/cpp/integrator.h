#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <functional>
#include <Eigen/Core>
#include "pseudoparticlestate.h"

class PseudoParticleState;

// used to integrate properties of the particle along the trajectory
// R is the type of the integration result
//template <class R>
class TrajectoryIntegrator {
    public:
        virtual double integrate(trajectory_t trajectory) const = 0;
};

class LinearIntegrator : public TrajectoryIntegrator {
    private:
        std::function<double(const Eigen::VectorXd&)> _callback;

    public:
        double integrate(trajectory_t trajectory) const;
        LinearIntegrator(std::function<double(const Eigen::VectorXd&)> callback);
};

// used to integrate properties of the particle along the trajectory
// R is the type of the integration result
//template <class R>
class TrajectoryLiveIntegrator {
    public:
        virtual void integrate(const PseudoParticleState& particle_state, double timestep) = 0;
        virtual double value() const = 0;
};

class LinearLiveIntegrator : public TrajectoryLiveIntegrator {//<double> {
    private:
        double _value;
        std::function<double(const Eigen::VectorXd&)> _callback;

    public:
        LinearLiveIntegrator(std::function<double(const Eigen::VectorXd&)> callback);
        void integrate(const PseudoParticleState& particle_state, double timestep);
        double value() const;


};

#endif
