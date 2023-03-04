#include "loop.h"

int integration_loop(std::vector<Eigen::VectorXd> &observations, double *t, 
        Eigen::Map<Eigen::VectorXd> &x, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, split_call_t split, pcg32::state_type seed,
        /*rng_call_t rng,*/ double timestep, 
        const std::vector<double>& t_observe, std::vector<double> &split_times,
        std::vector<Eigen::VectorXd> &split_points, const std::string& scheme_name){

    // the call signature is large because I think the effort of zipping
    // stuff into strucs/classes is large because of the intention to 
    // interface this with cython. (It would require all (data)classes
    // to be redefined in cython as well)
    //
    // assumes t_observe is sorted ascending

    int ndim = x.rows();
    
    scheme_t scheme_call = scheme_registry_lookup(scheme_name);

    auto rng = pcg32(seed);
    auto dist = std::normal_distribution<double>(0.0);

    Eigen::VectorXd x_new(ndim);
    Eigen::VectorXd rndvec(ndim);

    // initialize last_split store
    // last_split is equiv. to the particle spawn location at the start
    Eigen::VectorXd x_last_split(ndim);
    x_last_split = x;
    double t_last_split = *t;

    observations.reserve(t_observe.size());

    auto observe_it = t_observe.begin();
    int boundary_state = -1;
    while (observe_it != t_observe.end()){
        // boundary
        boundary_state = boundary(*t, x.data());
        if (boundary_state) break;

        //rng(rndvec, ndim); 
        for (int i = 0; i < ndim; i++){
            rndvec(i) = dist(rng);
        }

        // propagation
        *t = scheme_call(x_new, *t, x, rndvec, timestep, drift, diffusion);
        x = x_new;

        // observation
        while (*t >= *observe_it && observe_it != t_observe.end()){
            observations.push_back(x);
            observe_it++;
        }

        // splitting
        if (split(*t, x.data(), t_last_split, x_last_split.data())){
            x_last_split = x;
            t_last_split = *t;
            split_points.push_back(x);
            split_times.push_back(*t);
        }
    }

    for (auto& vec : split_points){
        std::cout << vec << "\n";
    }

    for (auto& t : split_times){
        std::cout << t << "\n";
    }

    return boundary_state;
}

int integration_loop_p(double *observations, int *observation_count, double *t,
        Eigen::Map<Eigen::VectorXd> &x, coeff_call_t drift, coeff_call_t diffusion,
        boundary_call_t boundary, split_call_t split, pcg32::state_type seed,
        /*rng_call_t rng,*/ double timestep, 
        const double *t_observe, int t_observe_count, int *split_count, double **split_times,
        double **split_points, const std::string& scheme_name){

    std::vector<Eigen::VectorXd> obs_vec;
    std::vector<double> t_obs_vec;
    std::vector<Eigen::VectorXd> split_points_vec;
    std::vector<double> split_times_vec;

    for (int i = 0; i < t_observe_count; i++){
       t_obs_vec.push_back(t_observe[i]);
    }

    int boundary_state = integration_loop(obs_vec, t, x, drift, diffusion, boundary, split, seed, timestep, t_obs_vec, split_times_vec, split_points_vec, scheme_name);

    // unneccessary: fewer observations are expected, more cannot happen
    //if (t_observe_count != obs_vec.size()){
    //    std::cout << "Warning: observation count mismatch!\n";
    //}

    int ndim = x.rows();
    int i = 0;
    for (auto& vec : obs_vec){
        for (int j = 0; j < ndim; j++){
            observations[i * ndim + j] = vec(j);
        }
        i++;
    }

    if (split_points_vec.size() != split_times_vec.size()){
        std::cout << "Error: count of split_points does not match count of split_times\n";
    }

    *split_points = new double[split_points_vec.size() * ndim];
    *split_times = new double[split_points_vec.size()];
    
    int k = 0;
    for (auto& vec : split_points_vec){
        for (int m = 0; m < ndim; m++){
            (*split_points)[k * ndim + m] = vec(m);
        }
        k++;
    }

    int a = 0;
    for (auto& t : split_times_vec){
        (*split_times)[a++] = t;
    }
    
    *split_count = split_points_vec.size();
    *observation_count = obs_vec.size();

    return boundary_state;
}

