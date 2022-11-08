#include "scheme.h"

double scheme_euler(Eigen::VectorXd& x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion) {
    int ndim = x.rows();
    Eigen::VectorXd drift_buf(ndim);
    Eigen::MatrixXd diffusion_buf(ndim, ndim);

    drift(drift_buf.data(), t, x.data());
    diffusion(diffusion_buf.data(), t, x.data());

    x_out = x + timestep * drift_buf + diffusion_buf * rndvec * sqrt(timestep);
    return t + timestep;
}


scheme_t scheme_registry_lookup(const std::string& name){
    std::map<std::string, scheme_t> s_map;

    s_map["euler"] = scheme_euler;
    //s_map.add("other_scheme", other_scheme);
    
    return s_map[name];
}
