#include "batch_kruells.h"

// msa = map_safe_access
double msa(std::map<std::string, double> map, std::string key){
    auto value = map.find(key);
    if (value != map.end()){
        return value->second;
    }else{
        throw std::invalid_argument("Parameter " + key + " not given");
        return 0;
    }
}


// ******************************************   KRUELLS 1 ****************************************** //

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

BatchKruells1::BatchKruells1(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells921_drift, _1, msa(params, "Tesc"));
    auto call_diffusion = std::bind(kruells921_diffusion, _1, msa(params, "Tesc"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "p0");
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = 0.001;
    opt.tracked = false;

    // initialize
    initialize(msa(params, "N"), callbacks, start, opt);
}

BatchKruells1::~BatchKruells1(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// **** SHOCK ACCELLERATION **** 

// shock coefficients
double a_from_shockparam(double beta_s, double r){
    return beta_s / 2 * (1 + 1 / r);
}

double b_from_shockparam(double beta_s, double r){
    return a_from_shockparam(beta_s, r) * (r - 1) / (r + 1);
}

// STREAMING VELOCITY
inline double kruells94_beta(double x, double Xsh, double a, double b){
    return a - b * std::tanh(x / Xsh);
}

inline double kruells94_dbetadx(double x, double Xsh, double b){
    double csh = std::cosh(x / Xsh);
    if (csh == HUGE_VAL){
        return 0;
    }else{
        return -b / (Xsh * std::pow(csh, 2));
    }
}

// SPATIAL DIFFUSION (proportional to beta)
inline double kruells94_kappa_dep(double x, double Xsh, double a, double b, double q){
    return q * pow(kruells94_beta(x, Xsh, a, b), 2);
}

inline double kruells94_dkappadx_dep(double x, double Xsh, double a, double b, double q){
    return 2 * q * kruells94_beta(x, Xsh, a, b) * kruells94_dbetadx(x, Xsh, b);
}

// dxs is also called X_sh
Eigen::VectorXd kruells_shockaccel_drift_94(const Eigen::VectorXd& x, double Xsh, double a, double b, double k_syn){
    Eigen::VectorXd v(2);
    v(0) = kruells94_beta(x(0), Xsh, a, b);
    v(1) = -(x(1)) * (kruells94_dbetadx(x(0), Xsh, b) / 3 + k_syn * sqrt(1 + std::pow(x(1), 2)));
    return v;
}

Eigen::VectorXd kruells_shockaccel_drift_92(const Eigen::VectorXd& x, double Xsh, double a, double b, double k_syn){
    Eigen::VectorXd v(2);
    v(0) = kruells94_beta(x(0), Xsh, a, b);
    v(1) = -(kruells94_dbetadx(x(0), Xsh, b) / 3 + k_syn * sqrt(1 + std::exp(2 * x(1))));
    return v;
}

// Kpar is also called kappa
Eigen::MatrixXd kruells_shockaccel_diffusion(const Eigen::VectorXd& x, double kappa){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = std::sqrt(2 * kappa);
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}


Eigen::VectorXd kruells_shockaccel2_drift_94(const Eigen::VectorXd& x, double Xsh, double a, double b, double k_syn, double q){
    Eigen::VectorXd v(2);
    v(0) = kruells94_dkappadx_dep(x(0), Xsh, a, b, q) + kruells94_beta(x(0), Xsh, a, b);
    v(1) = - (x(1)) * (kruells94_dbetadx(x(0), Xsh, b) / 3 + k_syn * sqrt(1 + std::pow(v(1), 2)));
    return v;
}

Eigen::VectorXd kruells_shockaccel2_drift_92(const Eigen::VectorXd& x, double Xsh, double a, double b, double k_syn, double q){
    Eigen::VectorXd v(2);
    v(0) = kruells94_dkappadx_dep(x(0), Xsh, a, b, q) + kruells94_beta(x(0), Xsh, a, b);
    double exponential = exp(2 * x(1));
    if (exponential == HUGE_VAL || exponential == HUGE_VALL || exponential == HUGE_VALF){
        std::cout << "huge val in exp 2";
        exponential = 0;
    }
    v(1) = -(kruells94_dbetadx(x(0), Xsh, b) / 3 + k_syn * sqrt(1 + exponential));
    return v;
}

Eigen::MatrixXd kruells_shockaccel2_diffusion(const Eigen::VectorXd& x, double Xsh, double a, double b, double q){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = sqrt(2 * kruells94_kappa_dep(x(0), Xsh, a, b, q));
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}
// ******************************************   KRUELLS 2 ****************************************** //

BatchKruells2::BatchKruells2(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel_drift_92, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"));
    auto call_diffusion = std::bind(kruells_shockaccel_diffusion, _1, msa(params, "kappa"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "u0");
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;

    // initialize
    initialize(msa(params, "N"), callbacks, start, opt);
}

BatchKruells2::~BatchKruells2(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 3 ****************************************** //

BatchKruells3::BatchKruells3(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel_drift_94, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"));
    auto call_diffusion = std::bind(kruells_shockaccel_diffusion, _1, msa(params, "kappa"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting points
    std::vector<SpaceTimePoint> starts;
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "y0");
    for (double t = 0; t <= msa(params, "Tmax"); t += msa(params, "t_inj")){
        starts.push_back(SpaceTimePoint(t, start_x));
    }

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;

    // initialize
    initialize(callbacks, starts, opt);
}

BatchKruells3::~BatchKruells3(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}



// ******************************************   KRUELLS 4 ****************************************** //


double kruells94_injection_region(const Eigen::VectorXd& s, double a_inj, double r_inj){
    if (s(0) < a_inj && s(0) > -a_inj) return r_inj;
    else return 0;
}

BatchKruells4::BatchKruells4(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel_drift_94, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"));
    auto call_diffusion = std::bind(kruells_shockaccel_diffusion, _1, msa(params, "kappa"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // integrator
    auto call_integrate = std::bind(kruells94_injection_region, _1, msa(params, "a_inj"), msa(params, "r_inj"));
    LinearLiveIntegrator lin_int{call_integrate};

    // starting points
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "y0");
    SpaceTimePoint start(0, start_x);

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;
    opt.add_integrator(lin_int);

    // initialize
    initialize(msa(params, "N"), callbacks, start, opt);
}

BatchKruells4::~BatchKruells4(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}


// ******************************************   KRUELLS 5 ****************************************** //

BatchKruells5::BatchKruells5(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel2_drift_94, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting points
    std::vector<SpaceTimePoint> starts;
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "y0");
    for (double t = 0; t <= msa(params, "Tmax"); t += msa(params, "t_inj")){
        starts.push_back(SpaceTimePoint(t, start_x));
    }

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;

    // initialize
    initialize(callbacks, starts, opt);
}

BatchKruells5::~BatchKruells5(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 6 ****************************************** //

BatchKruells6::BatchKruells6(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel_drift_92, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"));
    auto call_diffusion = std::bind(kruells_shockaccel_diffusion, _1, msa(params, "kappa"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting points
    std::vector<SpaceTimePoint> starts;
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "u0");
    for (double t = 0; t <= msa(params, "Tmax"); t += msa(params, "t_inj")){
        starts.push_back(SpaceTimePoint(t, start_x));
    }

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;

    // initialize
    initialize(callbacks, starts, opt);
}

BatchKruells6::~BatchKruells6(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 7 ****************************************** //

BatchKruells7::BatchKruells7(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel2_drift_92, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // starting points
    std::vector<SpaceTimePoint> starts;
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "u0");
    for (double t = 0; t <= msa(params, "Tmax"); t += msa(params, "t_inj")){
        starts.push_back(SpaceTimePoint(t, start_x));
    }

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;

    // initialize
    initialize(callbacks, starts, opt);
}

BatchKruells7::~BatchKruells7(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 8 ****************************************** //

BatchKruells8::BatchKruells8(std::map<std::string, double> params){
    // get a random generator
    std::random_device rdseed;
    pcg32::state_type seed = rdseed();
    _process = new WienerProcess(2, &seed);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel2_drift_94, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    PseudoParticleCallbacks callbacks{call_drift, call_diffusion};

    // integrator
    auto call_integrate = std::bind(kruells94_injection_region, _1, msa(params, "a_inj"), msa(params, "r_inj"));
    LinearLiveIntegrator lin_int{call_integrate};

    // starting points
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "y0");
    SpaceTimePoint start(0, start_x);

    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.process = _process;
    opt.timestep = msa(params, "dt");
    opt.tracked = false;
    opt.add_integrator(lin_int);

    // initialize
    initialize(msa(params, "N"), callbacks, start, opt);
}

BatchKruells8::~BatchKruells8(){
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

