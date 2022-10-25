#include "loop.h"

void ploop(std::vector<Eigen::VectorXd>& observations, double t0, 
        const Eigen::VectorXd& x0, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, rng_call_t rng, double timestep, 
        const std::vector<double>& t_observe, const std::string& scheme_name){

    // the call signature is large because I think the effort of zipping
    // stuff into strucs/classes is large because of the intention to 
    // interface this with cython. (It would require all (data)classes
    // to be redefined in cython as well)
    //
    // assumes t_observe is sorted ascending

    int ndim = x0.rows();
    
    scheme_t scheme_call = scheme_registry_lookup(scheme_name);

    double t = t0;
    Eigen::VectorXd x(ndim);
    Eigen::VectorXd x_new(ndim);
    Eigen::VectorXd rndvec(ndim);

    observations.reserve(t_observe.size());

    auto observe_it = t_observe.begin();
    int boundary_state;
    while (observe_it != t_observe.end()){
        boundary_state = boundary(t, x);
        if (boundary_state) break;

        rng(rndvec, ndim); 

        t = scheme_call(x_new, t, x, rndvec, timestep, drift, diffusion);
        x = x_new;

        if (t >= *observe_it){
            observations.push_back(x);
            observe_it++;
        }
    }
}

void ploop_pointer(double *observations, double t0, 
        const Eigen::VectorXd& x0, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, rng_call_t rng, double timestep, 
        const double *t_observe, int t_observe_count, const std::string& scheme_name){

    std::vector<Eigen::VectorXd> obs_vec;
    std::vector<double> t_obs_vec;

    for (int i = 0; i < t_observe_count; i++){
       t_obs_vec.push_back(t_observe[i]);
    }

    ploop(obs_vec, t0, x0, drift, diffusion, boundary, rng, timestep, t_obs_vec, scheme_name);

    if (t_observe_count != obs_vec.size()){
        std::cout << "Warning: observation count mismatch!\n";
    }

    int ndim = x0.rows();
    int i = 0;
    //double *observations = new double[ndim * obs_vec.size()];
    for (auto& vec : obs_vec){
        for (int j = 0; j < ndim; j++){
            observations[i * ndim + j] = vec(j);
        }
        i++;
    }

    //return observations;
}

