#ifndef LOOP_H
#define LOOP_H

#include <Eigen/Core>
#include <random>

#include "pcg/pcg_random.hpp"
#include "scheme.h"

typedef void (*rng_call_t)(Eigen::VectorXd& x_out, int ndim); // or similar
typedef int (*boundary_call_t)(double t, const Eigen::VectorXd& x);

int ploop_pointer(double *observations, double *t, 
        Eigen::Map<Eigen::VectorXd>& x, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, pcg32::state_type seed,/*rng_call_t rng,*/ double timestep, 
        const double *t_observe, int t_observe_count, const std::string& scheme_name);

int ploop(std::vector<Eigen::VectorXd>& observations, double *t, 
        Eigen::Map<Eigen::VectorXd>& x, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, pcg32::state_type seed,/*rng_call_t rng,*/ double timestep, 
        const std::vector<double>& t_observe, const std::string& scheme_name);

#endif
