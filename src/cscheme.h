#ifndef CSCHEME_H
#define CSCHEME_H

#include <Eigen/Core>
#include <iostream>

typedef struct SpaceTimePoint {
    double t;
    Eigen::VectorXd x;

    SpaceTimePoint();
    SpaceTimePoint(double t, const Eigen::VectorXd& x);
    operator std::string() const;
} SpaceTimePoint;

//typedef std::function<Eigen::VectorXd(const SpaceTimePoint&)> drift_t;
//typedef std::function<Eigen::MatrixXd(const SpaceTimePoint&)> diffusion_t;
typedef void (*coeff_call_t)(double *out, double t, const double *x);

SpaceTimePoint scheme_euler(const SpaceTimePoint& p, const Eigen::Map<Eigen::VectorXd>& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion);

#endif
