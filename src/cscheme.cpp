#include "cscheme.h"


SpaceTimePoint::SpaceTimePoint(double t, const Eigen::VectorXd& x) :
    t(t), x(x) {}

SpaceTimePoint::SpaceTimePoint() {}

SpaceTimePoint::operator std::string() const {
    std::ostringstream s;
    s << "(t|x) = (" << t << "|" << x << ")";
    return s.str();
}


SpaceTimePoint scheme_euler(const SpaceTimePoint& p, const Eigen::Map<Eigen::VectorXd>& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion) {
    //std::cout << "1\n";
    //std::cout << p.x << "\n";
    //std::cout << "2\n";
    //std::cout << rndvec << "\n";
    //std::cout << "3\n";
    //std::cout << &drift << "\n";
    //std::cout << "3 1/2\n";

    int ndim = p.x.rows();
    Eigen::VectorXd drift_buf(ndim);
    Eigen::MatrixXd diffusion_buf(ndim, ndim);
    //std::cout << "4\n";

    drift(drift_buf.data(), p.t, p.x.data());
    //std::cout << "5\n";
    //std::cout << drift_buf << "\n";
    diffusion(diffusion_buf.data(), p.t, p.x.data());
    //std::cout << "6\n";
    //std::cout << diffusion_buf << "\n";
    //std::cout << "7\n";

    Eigen::VectorXd x_new = p.x + timestep * drift_buf + diffusion_buf * rndvec * sqrt(timestep);
    return SpaceTimePoint(p.t + timestep, x_new);
}
