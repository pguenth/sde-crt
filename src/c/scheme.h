#ifndef SCHEME_H
#define SCHEME_H

#include <Eigen/Core>
#include <iostream>
#include <map>
#include <string>

//typedef std::function<Eigen::VectorXd(const SpaceTimePoint&)> drift_t;
//typedef std::function<Eigen::MatrixXd(const SpaceTimePoint&)> diffusion_t;
typedef void (*coeff_call_t)(double *out, double t, const double *x);
typedef double (*scheme_t)(Eigen::VectorXd& x_out, double t, const Eigen::Map<Eigen::VectorXd> &x,
                           const Eigen::VectorXd& rndvec, double timestep,
                           coeff_call_t drift, coeff_call_t diffusion);

double scheme_euler(Eigen::VectorXd &x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion);

scheme_t scheme_registry_lookup(const std::string& name);

#endif
