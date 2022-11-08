#ifndef SCHEME_H
#define SCHEME_H

#include <functional>
#include <Eigen/Core>
#include "pseudoparticlestate.h"
#include "stochasticprocess.h"
#include "broyden.h"


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
// Achterberg/Schure implements it only for x and not for p (I think, as they don't do momentum diff.)
// !!! watch out when using a diffusion coefficient that couples the equation, i.e. has off-diagonal terms
// then this implementation is probably not exactly the proposed one.
// probably B_yy also has to be zero
class KPPCScheme : public SDESchemeCopyable<KPPCScheme> {
    public:
        KPPCScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~KPPCScheme() = default;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;
};

// The implicit euler scheme from Kloeden/Platen 1992 Chapter 12.2 
// !!! watch out when using a diffusion coefficient that couples the equation, i.e. has off-diagonal terms
// then this implementation is probably not exactly the proposed one.
// probably B_yy also has to be zero
class ImplicitEulerScheme : public SDESchemeCopyable<ImplicitEulerScheme> {
    public:
        ImplicitEulerScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~ImplicitEulerScheme() = default;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;

    private:
        Eigen::VectorXd get_implicit(Eigen::VectorXd x, const SpaceTimePoint& p, double timestep, Eigen::VectorXd rnd) const; 
};

// Kloeden/Platen Num.Sol.Stoch.Diff.Eq. ch. 15.1. eq. 1.3
// errors in the calculation of the RMS values
// !!! does only work for diffusion matrices with only the xx-component being nonzero.
class SecondOrderScheme : public SDESchemeCopyable<SecondOrderScheme> {
    public:
        SecondOrderScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~SecondOrderScheme() = default;
        //virtual Scheme *copy(std::vector<uint64_t> seeds) const;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;
};

// Kloeden/Platen Num.Sol.Stoch.Diff.Eq. ch. 15.1. eq. 1.3
// errors in the calculation of the RMS values
// !!! does only work for diffusion matrices with only the xx-component being nonzero.
// not sure of the vector generalization in this scheme (in comparison to SecondOrderScheme)
class SecondOrderScheme2 : public SDESchemeCopyable<SecondOrderScheme2> {
    public:
        SecondOrderScheme2(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~SecondOrderScheme2() = default;
        //virtual Scheme *copy(std::vector<uint64_t> seeds) const;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;
};

// See Gardiner: Stochastic Methods Handbook for Nat. and Soc. sciences: 15.5.6
// This one seems to have a bug in get_implicit: the drift term is not multiplied by the timestep.
// Fixed in SemiImplicitWeakScheme2 while this one is kept because results depend on it
class SemiImplicitWeakScheme : public SDESchemeCopyable<SemiImplicitWeakScheme> {
    public:
        SemiImplicitWeakScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~SemiImplicitWeakScheme() = default;
        //virtual Scheme *copy(std::vector<uint64_t> seeds) const;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;

    protected:
        // this gets the result of: 2x - 2x_bar + A(p_bar) - 1/2 C(p_bar) + B(p_bar) * dW
        // which we need to find the roots of to get the desired x_bar
        virtual Eigen::VectorXd get_implicit(Eigen::VectorXd x_bar, const SpaceTimePoint& p, double timestep, Eigen::VectorXd rnd) const; 

        // C_i,j,j from eq. 15.5.24 summed over j, given as vector in i
        Eigen::VectorXd get_C(const SpaceTimePoint& p) const; 

        // gets a vector of matrices (basically a tensor) where each vector element is the diffusion
        // matrix differentiated in the respective direction.
        // one may call it "gradient of a matrix field"
        std::vector<Eigen::MatrixXd> get_B_diff(const SpaceTimePoint& p, double delta /*= 0.00001*/) const;
};

// See Gardiner: Stochastic Methods Handbook for Nat. and Soc. sciences: 15.5.6
// and see SemiImplicitWeakScheme
// this one fixes the missing timesteps in SemiImplicitWeakScheme
class SemiImplicitWeakScheme2 : public SDESchemeCopyable<SemiImplicitWeakScheme2> {
        //
    public:
        SemiImplicitWeakScheme2(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process);
        virtual ~SemiImplicitWeakScheme2() = default;
    
        //virtual Scheme *copy(std::vector<uint64_t> seeds) const;
        virtual SpaceTimePoint propagate(const SpaceTimePoint& p) const;

    protected:
        // this gets the result of: 2x - 2x_bar + A(p_bar) * dt - 1/2 C(p_bar) * dt + B(p_bar) * dW
        // which we need to find the roots of to get the desired x_bar
        Eigen::VectorXd get_implicit(Eigen::VectorXd x_bar, const SpaceTimePoint& p, double timestep, Eigen::VectorXd rnd) const; 

        // C_i,j,j from eq. 15.5.24 summed over j, given as vector in i
        Eigen::VectorXd get_C(const SpaceTimePoint& p) const; 

        // gets a vector of matrices (basically a tensor) where each vector element is the diffusion
        // matrix differentiated in the respective direction.
        // one may call it "gradient of a matrix field"
        std::vector<Eigen::MatrixXd> get_B_diff(const SpaceTimePoint& p, double delta /*= 0.00001*/) const;
};

#endif
