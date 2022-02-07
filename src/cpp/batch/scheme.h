#ifndef SCHEME_H
#define SCHEME_H

#include <functional>
#include <Eigen/Core>
#include "pseudoparticlestate.h"
#include "stochasticprocess.h"


class Scheme {
    public:
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const = 0;
        virtual double timestep_at(const SpaceTimePoint& p) const = 0;
        virtual Scheme *copy(std::vector<uint64_t> seeds) const = 0;
};

typedef std::function<Eigen::VectorXd(const SpaceTimePoint&)> drift_t;
typedef std::function<Eigen::MatrixXd(const SpaceTimePoint&)> diffusion_t;
typedef std::function<double(const SpaceTimePoint&)> timestep_t;

class SDEScheme : public Scheme {
    protected:
        drift_t _drift;
        diffusion_t _diffusion;
        timestep_t _timestep;
        StochasticProcess *_process;
        std::vector<uint64_t> seeds;

    public:
        SDEScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual double timestep_at(const SpaceTimePoint& p) const;
        virtual Scheme *copy(std::vector<uint64_t> seeds) const = 0;
        Eigen::VectorXd next_random(const SpaceTimePoint& p) const;
        Eigen::VectorXd next_random(double timestep) const;
};

class EulerScheme : public SDEScheme {
    public:
        EulerScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual Scheme *copy(std::vector<uint64_t> seeds) const;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;
};

#endif
