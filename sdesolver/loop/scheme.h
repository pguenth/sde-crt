#ifndef SCHEME_H
#define SCHEME_H

#include <Eigen/Core>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "broyden.h"

/**
 * Callback type for SDE coefficients.
 * @param out array (must be allocated with at least ndim (drift) or
 *      ndim x ndim (diffusion) fields) to which the result is written
 * @param t Time of the solution at which the coefficient should be evaluated
 * @param x Position of the solution at which the coefficient should be
 *      evaluated
 */
typedef void (*coeff_call_t)(double *out, double t, const double *x);

/**
 * Callback type for a scheme
 */
typedef double (*scheme_t)(Eigen::VectorXd& x_out, double t, const Eigen::Map<Eigen::VectorXd> &x,
                           const Eigen::VectorXd& rndvec, double timestep,
                           coeff_call_t drift, coeff_call_t diffusion);

/**
 * Simple euler scheme
 */
double scheme_euler(Eigen::VectorXd &x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion);

/**
 * KPPC
 */
double scheme_kppc(Eigen::VectorXd &x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion);

/**
 * Semiimplicit-Weak second order 
 */
double scheme_semiimplicit_weak(Eigen::VectorXd &x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion);

/**
 * This method takes a string and returns a corresponding scheme callback.
 * This is to enable the control of the scheme from high-level python 
 * without having to expose the scheme callback itself through cython.
 *
 * Currently supported:
 *  - 'euler' The simple euler scheme
 *  - 'kppc' Kloeden-Platen predictor-corrector scheme (Achterberg 2011)
 *  - 'semiimplicit_weak' Second-order semiimplicit weak scheme (Gardiner)
 */
scheme_t scheme_registry_lookup(const std::string& name);

#endif
