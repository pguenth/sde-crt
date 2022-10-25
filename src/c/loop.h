#ifndef LOOP_H
#define LOOP_H

#include <Eigen/Core>

#include "scheme.h"

typedef void (*rng_call_t)(Eigen::VectorXd& x_out, int ndim); // or similar
typedef int (*boundary_call_t)(double t, const Eigen::VectorXd& x);

void ploop_pointer(double *observations, double t0, 
        const Eigen::VectorXd& x0, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, rng_call_t rng, double timestep, 
        const double *t_observe, int t_observe_count, const std::string& scheme_name);

void ploop(std::vector<Eigen::VectorXd>& observations, double t0, 
        const Eigen::VectorXd& x0, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, rng_call_t rng, double timestep, 
        const std::vector<double>& t_observe, const std::string& scheme_name);

#endif
