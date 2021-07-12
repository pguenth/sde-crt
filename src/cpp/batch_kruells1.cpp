#include "batch_kruells1.h"

inline double kruells1_kappa(double x, double y){
    return 1;
}

inline double kruells1_dkappa_dx(double x, double y){
    return 0;
}

inline double kruells1_beta(double x, double Xsh, double a, double b){
    return a - b * tanh(x / Xsh);

}

inline double kruells1_dbeta_dx(double x, double Xsh, double b){
    return -b / (Xsh * pow(cosh(x / Xsh), 2));

}


Eigen::VectorXd kruells1_drift(const Eigen::VectorXd& x, double Xsh, double a, double b){
    Eigen::VectorXd v(2);
    v(0) = kruells1_dkappa_dx(x(0), x(1)) + kruells1_beta(x(0), Xsh, a, b);
    v(1) = -(x(1)) * kruells1_dbeta_dx(x(0), Xsh, b) / 3.0;
    return v;
}

Eigen::MatrixXd kruells1_diffusion(const Eigen::VectorXd& x, double Xsh, double a, double b){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = sqrt(2 * kruells1_kappa(x(0), x(1)));
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}

BatchKruells1::BatchKruells1(double x0, double y0, int N, double Tmax, double L, double Xsh, double a, double b){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(Tmax);

    // spatial breakpoint
    Eigen::VectorXd xmin(2), xmax(2);
    xmin << -L, 0;
    xmax << L, 1000;
    _slimit = new BreakpointSpatial(xmin, xmax);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells1_drift, _1, Xsh, a, b);
    auto call_diffusion = std::bind(kruells1_diffusion, _1, Xsh, a, b);
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << x0, y0;
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = 0.001;

    // initialize
    initialize(N, callbacks, start, opt);
}

BatchKruells1::~BatchKruells1(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

//std::vector<double> BatchSourcetest::integrate(){
//    return apply_integrator(*_sintegrator);
//}
