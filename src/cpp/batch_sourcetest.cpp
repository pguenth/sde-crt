#include "batch_sourcetest.h"

double sourcetest_integrate(const Eigen::VectorXd& x){
    Eigen::VectorXd x_new = x.unaryExpr([](double x){
            if (x >= -0.5 && x <= 0.0) return 2.0;
            else return 0.0;
        });
    return x_new[0];
}

Eigen::VectorXd sourcetest_drift(const Eigen::VectorXd& x){
    Eigen::VectorXd v(1);
    v << 0;
    return v;
}

Eigen::MatrixXd sourcetest_diffusion(const Eigen::VectorXd& x){
    Eigen::MatrixXd v(1, 1);
    v << 1;
    return v;
}

BatchSourcetest::BatchSourcetest(double x0, int N, double Tmax, double x_min, double x_max){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(1, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(Tmax);

    // spatial breakpoint
    Eigen::VectorXd xmin(1), xmax(1);
    xmin << x_min;
    xmax << x_max;
    _slimit = new BreakpointSpatial(xmin, xmax);

    // integrator
    _sintegrator = new LinearIntegrator(sourcetest_integrate);

    // callbacks
    // not sure if &function is better
    PseudoParticleCallbacks callbacks{sourcetest_drift, sourcetest_diffusion};

    // starting point
    Eigen::VectorXd start_x(1);
    start_x << x0;
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = 0.005;

    // initialize
    initialize(N, callbacks, start, opt);
}

BatchSourcetest::~BatchSourcetest(){
    delete _process;
    delete _tlimit;
    delete _slimit;
    delete _sintegrator;
}

std::vector<double> BatchSourcetest::integrate(){
    return apply_integrator(*_sintegrator);
}
