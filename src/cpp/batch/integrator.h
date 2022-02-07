#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <functional>
#include <Eigen/Core>
#include "pseudoparticlestate.h"

class SpaceTimePoint;

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
        virtual void integrate(const SpaceTimePoint& p, double timestep) = 0;
        virtual double value() const = 0;
        virtual TrajectoryLiveIntegrator* clone() const = 0;
};

//https://stackoverflow.com/questions/5731217/how-to-copy-create-derived-class-instance-from-a-pointer-to-a-polymorphic-base-c
//CRTP idiom

template <class D>
class TrajectoryLiveIntegratorB : public TrajectoryLiveIntegrator {
    public:
        virtual TrajectoryLiveIntegrator* clone() const {
            return new D(static_cast<const D&>(*this));
        }
};

class LinearLiveIntegrator : public TrajectoryLiveIntegratorB<LinearLiveIntegrator> {//<double> {
    private:
        double _value;
        std::function<double(const Eigen::VectorXd&)> _callback;

    public:
        LinearLiveIntegrator(std::function<double(const Eigen::VectorXd&)> callback);
        void integrate(const SpaceTimePoint& p, double timestep);
        double value() const;


};

#endif
