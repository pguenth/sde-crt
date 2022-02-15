#ifndef SCHEME_H
#define SCHEME_H

#include <functional>
#include <Eigen/Core>
#include "pseudoparticlestate.h"
#include "stochasticprocess.h"


class Scheme {
    public:
        virtual ~Scheme() = default;
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
        virtual ~SDEScheme() = default;
        virtual double timestep_at(const SpaceTimePoint& p) const;
        virtual Scheme *copy(std::vector<uint64_t> seeds) const = 0;
        Eigen::VectorXd next_random() const;
};

//https://stackoverflow.com/questions/5731217/how-to-copy-create-derived-class-instance-from-a-pointer-to-a-polymorphic-base-c
//CRTP idiom

template <class D>
class SDESchemeCopyable : public SDEScheme {
    public:
        SDESchemeCopyable(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
            SDEScheme(drift, diffusion, timestep, process) {}
        virtual ~SDESchemeCopyable() = default;
        virtual Scheme* copy(std::vector<uint64_t> seeds) const {
            StochasticProcess *new_p = _process->copy(seeds);
            return new D(_drift, _diffusion, _timestep, new_p);
        }
};

class EulerScheme : public SDESchemeCopyable<EulerScheme> {
    public:
        EulerScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~EulerScheme() = default;
        //virtual Scheme *copy(std::vector<uint64_t> seeds) const;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;
};

// See https://arxiv.org/pdf/1103.3049.pdf and Kloeden/Platen 1992 Chapter 15.5 Order 2.0 weak pred.-corr. methods
// Achterberg/Schure implements it only for x and not for p (I think)
class KPPCScheme : public SDESchemeCopyable<KPPCScheme> {
    public:
        KPPCScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~KPPCScheme() = default;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;
};

#endif
