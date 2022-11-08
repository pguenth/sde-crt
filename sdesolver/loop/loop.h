#ifndef LOOP_H
#define LOOP_H

#include <Eigen/Core>
#include <random>

#include "pcg/pcg_random.hpp"
#include "scheme.h"

/**
 * Currently not used
 */
typedef void (*rng_call_t)(Eigen::VectorXd& x_out, int ndim); // or similar
                                                              //
/**
 * Callback type for boundary checks. The time and position of the 
 * solution is passed. Is expected to return 0 if no boundary is reached and
 * some other value if a boundary is reached. -1 is reserved as error state
 */
typedef int (*boundary_call_t)(double t, const double *x);

/**
 * variant of integration_loop with pointers to avoid having to convert
 * std::vector types in cython. eigen types would be possible, but would
 * probably lead to much more difficult to read code in integration_loop
 */
int integration_loop_p(double *observations, int *observation_count, double *t, 
        Eigen::Map<Eigen::VectorXd>& x, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, pcg32::state_type seed,/*rng_call_t rng,*/ double timestep, 
        const double *t_observe, int t_observe_count, const std::string& scheme_name);

/**
 * calculates one solution of a stochastic differential equation.
 *
 * @param observations Container to which the position (x) is written to when
 *      an observation time is reached. Those are written in the order of
 *      t_observe, which in itself is assumed to be sorted ascending.
 * @param t The start time of the integration. The final time will be written
 *      to this pointer. This is relevant if a boundary is reached.
 * @param x The start position of the integration. The final position will be
 *      written to this pointer. This is relevant if a boundary is reached.
 * @param drift Callback for the drift coefficient
 * @param diffusion Callback for the diffusion coefficient
 * @param boundary Callback for the boundary coefficient
 * @param seed The seed for the PCG32 random number generator that will be used
 *      for this solution.
 * @param timestep The desired timestep.
 * @param t_observe Container of times at which the solution's position should
 *      be stored in observations. The first position after reaching the
 *      respective time is stored. This vector must be sorted ascending.
 * @param scheme_name Name of the scheme to use. See scheme.h for further info.
 */
int integration_loop(std::vector<Eigen::VectorXd>& observations, double *t, 
        Eigen::Map<Eigen::VectorXd>& x, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, pcg32::state_type seed,/*rng_call_t rng,*/ double timestep, 
        const std::vector<double>& t_observe, const std::string& scheme_name);

#endif
