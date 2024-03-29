#include "batch_sourcetest.h"

double sourcetest_integrate(const Eigen::VectorXd& x){
    Eigen::VectorXd x_new = x.unaryExpr([](double x){
            if (x >= -0.5 && x <= 0.0) return 2.0;
            else return 0.0;
        });
    return x_new[0];
}

Eigen::VectorXd sourcetest_drift(const SpaceTimePoint& p){
    Eigen::VectorXd v(1);
    v << 0;
    return v;
}

Eigen::MatrixXd sourcetest_diffusion(const SpaceTimePoint& p){
    Eigen::MatrixXd v(1, 1);
    v << 1;
    return v;
}

double ts_const_2(const SpaceTimePoint& p, double dt){
    return dt;
}

double integrator_cb(const Eigen::VectorXd& x){
    return 0.01 * x(0);
}

BatchSourcetest::BatchSourcetest(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(1);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(params["Tmax"]);

    // spatial breakpoint
    Eigen::VectorXd xmin(1), xmax(1);
    xmin << params["x_min"];
    xmax << params["x_max"];
    _slimit = new BreakpointSpatial(xmin, xmax);

    // integrator
    _sintegrator = new LinearIntegrator(sourcetest_integrate);
    LinearLiveIntegrator liveintegrator(sourcetest_integrate);

    // callbacks
    // not sure if &function is better
    auto ts = std::bind(ts_const_2, std::placeholders::_1, 0.005);
    _scheme = new EulerScheme(sourcetest_drift, sourcetest_diffusion, ts, _process);

    // starting point
    Eigen::VectorXd start_x(1);
    start_x << params["x0"];
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    opt.breakpoints.push_back(_slimit);
    
    opt.add_integrator(liveintegrator);
    opt.tracked = false;
    opt.scheme = _scheme;

    // initialize
    initialize(params["N"], start, opt);
}

BatchSourcetest::~BatchSourcetest(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    delete _slimit;
    delete _sintegrator;
}

std::vector<double> BatchSourcetest::integrate(){
    return apply_integrator(*_sintegrator);
}
