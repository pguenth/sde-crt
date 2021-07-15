#include "batch_kruells92.h"

// Vs = V(x)|x<0
// r compression ratio
// --> a = Vs / 2 * (1 + 1/r)
// --> b = a * (r - 1) / (r + 1)
inline double kruells92_V(double x, double dxs, double a, double b){
    return a - b * tanh(x / dxs);

}

inline double kruells92_dVdx(double x, double dxs, double b){
    return -b / (dxs * pow(cosh(x / dxs), 2));

}

Eigen::VectorXd kruells921_drift(const Eigen::VectorXd& x, double T){
    Eigen::VectorXd v(2);
    v(0) = -x(0) / T;
    v(1) = 3 / (4 * T) - 0.00001 * sqrt(1 + exp(2 * v(1))) / T;
    return v;
}

Eigen::MatrixXd kruells921_diffusion(const Eigen::VectorXd& x, double T){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = 0;
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = sqrt(1 / (2 * T));
    return v;
}

BatchKruells921::BatchKruells921(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(params["Tmax"]);

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells921_drift, _1, params["Tesc"]);
    auto call_diffusion = std::bind(kruells921_diffusion, _1, params["Tesc"]);
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << params["x0"], params["p0"];
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = 0.001;
    opt.tracked = false;

    // initialize
    initialize(params["N"], callbacks, start, opt);
}

BatchKruells921::~BatchKruells921(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

Eigen::VectorXd kruells922_drift(const Eigen::VectorXd& x, double dxs, double a, double b, double beta_s){
    Eigen::VectorXd v(2);
    v(0) = kruells92_V(x(0), dxs, a, b);
    v(1) = - kruells92_dVdx(x(0), dxs, b) / 3 - beta_s * sqrt(1 + exp(2 * x(1)));
    return v;
}

Eigen::MatrixXd kruells922_diffusion(const Eigen::VectorXd& x, double Kpar){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = sqrt(2 * Kpar);
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}

BatchKruells922::BatchKruells922(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(params["Tmax"]);

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio

    double a = params["Vs"] / 2 * (1 + 1/params["r"]);
    double b = a * (params["r"] - 1) / (params["r"] + 1);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells922_drift, _1, params["dxs"], a, b, params["beta_s"]);
    auto call_diffusion = std::bind(kruells922_diffusion, _1, params["Kpar"]);
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << params["x0"], params["p0"];
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = params["dt"];
    opt.tracked = false;

    // initialize
    initialize(params["N"], callbacks, start, opt);
}

BatchKruells922::~BatchKruells922(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

BatchKruells923::BatchKruells923(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(params["Tmax"]);

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio

    double a = params["Vs"] / 2 * (1 + 1/params["r"]);
    double b = a * (params["r"] - 1) / (params["r"] + 1);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells922_drift, _1, params["dxs"], a, b, params["beta_s"]);
    auto call_diffusion = std::bind(kruells922_diffusion, _1, params["Kpar"]);
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting points
    std::vector<SpaceTimePoint> starts;
    Eigen::VectorXd start_x(2);
    start_x << params["x0"], params["p0"];
    for (double t = 0; t <= params["Tmax"]; t += params["r_inj"]){
        starts.push_back(SpaceTimePoint(t, start_x));
    }

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = params["dt"];
    opt.tracked = false;

    // initialize
    initialize(callbacks, starts, opt);
}

BatchKruells923::~BatchKruells923(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

double kruells924_injection_region(const Eigen::VectorXd& s, double r_inj, double dx_inj){
    if (s(0) < r_inj && s(0) > -r_inj) return dx_inj;
    else return 0;
}

BatchKruells924::BatchKruells924(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(params["Tmax"]);

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio

    double a = params["Vs"] / 2 * (1 + 1/params["r"]);
    double b = a * (params["r"] - 1) / (params["r"] + 1);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells922_drift, _1, params["dxs"], a, b, params["beta_s"]);
    auto call_diffusion = std::bind(kruells922_diffusion, _1, params["Kpar"]);
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // integrator
    auto call_integrate = std::bind(kruells924_injection_region, _1, params["r_inj"], params["dx_inj"]);
    LinearLiveIntegrator lin_int{call_integrate};

    // starting points
    Eigen::VectorXd start_x(2);
    start_x << params["x0"], params["p0"];
    SpaceTimePoint start(0, start_x);

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = params["dt"];
    opt.tracked = false;
    opt.add_integrator(lin_int);

    // initialize
    initialize(params["N"], callbacks, start, opt);
}

BatchKruells924::~BatchKruells924(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}
