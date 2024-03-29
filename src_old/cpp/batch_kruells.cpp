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

double ts_const(const SpaceTimePoint& p, double dt){
    return dt;
}

// ******************************************   KRUELLS 1 ****************************************** //

Eigen::VectorXd kruells921_drift(const SpaceTimePoint& p, double T){
    Eigen::VectorXd v(2);
    v(0) = -p.x(0) / T;
    v(1) = 3 / (4 * T) - 0.00001 * sqrt(1 + exp(2 * p.x(1))) / T;
    return v;
}

Eigen::MatrixXd kruells921_diffusion(const SpaceTimePoint& p, double T){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = 0;
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = sqrt(1 / (2 * T));
    return v;
}

BatchKruells1::BatchKruells1(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "p0");
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(msa(params, "N"), start, opt);
}

BatchKruells1::~BatchKruells1(){
    delete _scheme;
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


Eigen::VectorXd kruells_shockaccel_drift_92(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn){
    Eigen::VectorXd v(2);
    v(0) = kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = -(kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * sqrt(1 + std::exp(2 * p.x(1))));
    return v;
}

// dxs is also called X_sh
Eigen::VectorXd kruells_shockaccel_drift_94(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn){
    Eigen::VectorXd v(2);
    v(0) = kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = -(p.x(1)) * (kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * sqrt(1 + std::pow(p.x(1), 2)));
    return v;
}

// Kpar is also called kappa
Eigen::MatrixXd kruells_shockaccel_diffusion(const SpaceTimePoint& p, double kappa){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = std::sqrt(2 * kappa);
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}

// dxs is also called X_sh
Eigen::VectorXd kruells_shockaccel_pure_drift(const SpaceTimePoint& p, double Xsh, double a, double b){
    Eigen::VectorXd v(2);
    v(0) = kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = -(p.x(1)) * kruells94_dbetadx(p.x(0), Xsh, b) / 3;
    return v;
}


Eigen::VectorXd kruells_shockaccel2_drift_94(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn, double q){
    Eigen::VectorXd v(2);
    v(0) = kruells94_dkappadx_dep(p.x(0), Xsh, a, b, q) + kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = - (p.x(1)) * (kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * sqrt(1 + std::pow(p.x(1), 2)));
    return v;
}

Eigen::VectorXd kruells_shockaccel2_drift_94_2(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn, double q){
    Eigen::VectorXd v(2);
    v(0) = kruells94_dkappadx_dep(p.x(0), Xsh, a, b, q) + kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = - (p.x(1)) * (kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * p.x(1));
    //if (v(1) < 0)
    //    std::cout << "p.x(1) " << p.x(1) << ", dbetadx " << kruells94_dbetadx(p.x(0), Xsh, b) << ", k_syn " << k_syn << "\n";
    return v;
}

Eigen::VectorXd kruells_shockaccel2_drift_92(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn, double q){
    Eigen::VectorXd v(2);
    v(0) = kruells94_dkappadx_dep(p.x(0), Xsh, a, b, q) + kruells94_beta(p.x(0), Xsh, a, b);
    double exponential = exp(2 * p.x(1));
    if (exponential == HUGE_VAL || exponential == HUGE_VALL || exponential == HUGE_VALF){
        std::cout << "huge val in exp 2";
        exponential = 0;
    }
    v(1) = -(kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * sqrt(1 + exponential));
    return v;
}

Eigen::MatrixXd kruells_shockaccel2_diffusion(const SpaceTimePoint& p, double Xsh, double a, double b, double q){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = sqrt(2 * kruells94_kappa_dep(p.x(0), Xsh, a, b, q));
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}
// ******************************************   KRUELLS 2 ****************************************** //

BatchKruells2::BatchKruells2(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

    // starting point
    Eigen::VectorXd start_x(2);
    start_x << msa(params, "x0"), msa(params, "u0");
    SpaceTimePoint start{0, start_x};


    // register options
    PseudoParticleOptions opt;
    opt.breakpoints.push_back(_tlimit);
    //opt.breakpoints.push_back(_slimit);
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(msa(params, "N"), start, opt);
}

BatchKruells2::~BatchKruells2(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 3 ****************************************** //

BatchKruells3::BatchKruells3(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells3::~BatchKruells3(){
    delete _scheme;
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
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;
    opt.add_integrator(lin_int);

    // initialize
    initialize(msa(params, "N"), start, opt);
}

BatchKruells4::~BatchKruells4(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}


// ******************************************   KRUELLS 5 ****************************************** //

BatchKruells5::BatchKruells5(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells5::~BatchKruells5(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 6 ****************************************** //

BatchKruells6::BatchKruells6(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells6::~BatchKruells6(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 7 ****************************************** //

BatchKruells7::BatchKruells7(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells7::~BatchKruells7(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 8 ****************************************** //

BatchKruells8::BatchKruells8(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;
    opt.add_integrator(lin_int);

    // initialize
    initialize(msa(params, "N"), start, opt);
}

BatchKruells8::~BatchKruells8(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 9 ****************************************** //

BatchKruells9::BatchKruells9(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_2, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells9::~BatchKruells9(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 10 ****************************************** //

BatchKruells10::BatchKruells10(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    double L = msa(params, "L");
    Eigen::VectorXd xmin(2), xmax(2);
    xmin << -L, 0;
    xmax << L, std::numeric_limits<double>::infinity();
    _slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel_pure_drift, _1, msa(params, "Xsh"), a, b);
    auto call_diffusion = std::bind(kruells_shockaccel_diffusion, _1, msa(params, "kappa"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells10::~BatchKruells10(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 11 ****************************************** //

BatchKruells11::BatchKruells11(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    Eigen::VectorXd xmin(2), xmax(2);
    xmin << -msa(params, "L"), 0;
    xmax << msa(params, "L"), std::numeric_limits<double>::infinity();
    _slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "beta_s"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "beta_s"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_2, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.breakpoints.push_back(_slimit);
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells11::~BatchKruells11(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}


Eigen::VectorXd kruells_shockaccel2_drift_94_3(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn, double q, double Xdiff){
    Eigen::VectorXd v(2);
    v(0) = - kruells94_kappa_dep(p.x(0), Xsh, a, b, q) / Xdiff + kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = - (p.x(1)) * (kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * p.x(1));
    return v;
}

// ******************************************   KRUELLS 12 ****************************************** //
// Kruells 9 with dkappa/dx from eq. (19) = -kappa/Xdiff

BatchKruells12::BatchKruells12(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_3, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"), msa(params, "Xdiff"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells12::~BatchKruells12(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

Eigen::VectorXd kruells_shockaccel2_drift_94_4(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn, double q, double Xdiff){
    Eigen::VectorXd v(2);
    v(0) = kruells94_kappa_dep(p.x(0), Xsh, a, b, q) / Xdiff + kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = - (p.x(1)) * (kruells94_dbetadx(p.x(0), Xsh, b) / 3 + k_syn * p.x(1));
    return v;
}

// ******************************************   KRUELLS 12 ****************************************** //
// Kruells 9 with dkappa/dx from eq. (19) = kappa/Xdiff

BatchKruells13::BatchKruells13(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_4, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"), msa(params, "Xdiff"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells13::~BatchKruells13(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   KRUELLS 14 ****************************************** //
// Kruells 9 with semi-implicit scheme

BatchKruells14::BatchKruells14(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_2, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new SemiImplicitWeakScheme2(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells14::~BatchKruells14(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}
//
// ******************************************   KRUELLS 15 ****************************************** //
// Kruells 9 with fully implicit scheme

BatchKruells15::BatchKruells15(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_2, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new ImplicitEulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells15::~BatchKruells15(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

//
// ******************************************   KRUELLS 16 ****************************************** //
// Kruells 9 with KPPC

BatchKruells16::BatchKruells16(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

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
    auto call_drift = std::bind(kruells_shockaccel2_drift_94_2, _1, msa(params, "Xsh"), a, b, msa(params, "k_syn"), msa(params, "q"));
    auto call_diffusion = std::bind(kruells_shockaccel2_diffusion, _1, msa(params, "Xsh"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new KPPCScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchKruells16::~BatchKruells16(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// ******************************************   Achterberg 2011 1  ****************************************** //
// testing kappa = q * beta bzw. D = q * V in achterbergs notation
// parametrisierung wie kruells94, also nicht ln(p)
//
inline double achterberg_kappa(double x, double Ls, double a, double b, double q){
    return q * kruells94_beta(x, Ls, a, b);
}

inline double achterberg_dkappa_dx(double x, double Ls, double b, double q){
    return q * kruells94_dbetadx(x, Ls, b);
}

Eigen::VectorXd achterberg_drift(const SpaceTimePoint& p, double Ls, double a, double b, double q){
    Eigen::VectorXd v(2);
    v(0) = achterberg_dkappa_dx(p.x(0), Ls, b, q) + kruells94_beta(p.x(0), Ls, a, b);
    v(1) = - (p.x(1)) * (kruells94_dbetadx(p.x(0), Ls, b) / 3);
    return v;
}

Eigen::MatrixXd achterberg_diffusion(const SpaceTimePoint& p, double Ls, double a, double b, double q){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = sqrt(2 * achterberg_kappa(p.x(0), Ls, a, b, q));
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}

BatchAchterberg1::BatchAchterberg1(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg1::~BatchAchterberg1(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}


BatchAchterberg1KPPC::BatchAchterberg1KPPC(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new KPPCScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg1KPPC::~BatchAchterberg1KPPC(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// with ln(p) parametrisation like in the paper
Eigen::VectorXd achterberg_drift_ln(const SpaceTimePoint& p, double Ls, double a, double b, double q){
    Eigen::VectorXd v(2);
    v(0) = achterberg_dkappa_dx(p.x(0), Ls, b, q) + kruells94_beta(p.x(0), Ls, a, b);
    v(1) = -1 * (kruells94_dbetadx(p.x(0), Ls, b) / 3);
    return v;
}

BatchAchterberg2::BatchAchterberg2(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2::~BatchAchterberg2(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}
BatchAchterberg2KPPC::BatchAchterberg2KPPC(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new KPPCScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2KPPC::~BatchAchterberg2KPPC(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

BatchAchterberg2Implicit::BatchAchterberg2Implicit(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new ImplicitEulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2Implicit::~BatchAchterberg2Implicit(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

BatchAchterberg2SecondOrder::BatchAchterberg2SecondOrder(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new SecondOrderScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2SecondOrder::~BatchAchterberg2SecondOrder(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}
BatchAchterberg2SecondOrder2::BatchAchterberg2SecondOrder2(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new SecondOrderScheme2(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2SecondOrder2::~BatchAchterberg2SecondOrder2(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

BatchAchterberg2SemiImplicit::BatchAchterberg2SemiImplicit(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new SemiImplicitWeakScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2SemiImplicit::~BatchAchterberg2SemiImplicit(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

BatchAchterberg2SemiImplicit2::BatchAchterberg2SemiImplicit2(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(msa(params, "V"), msa(params, "r"));
    double b = b_from_shockparam(msa(params, "V"), msa(params, "r"));

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(achterberg_drift_ln, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_diffusion = std::bind(achterberg_diffusion, _1, msa(params, "Ls"), a, b, msa(params, "q"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new SemiImplicitWeakScheme2(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}

BatchAchterberg2SemiImplicit2::~BatchAchterberg2SemiImplicit2(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// *************** KRUELLS B1 *************
// Schlumpfhüte: KruellsBx
//


Eigen::VectorXd kruells_shockaccel3_drift_94(const SpaceTimePoint& p, double Xsh, double a, double b, double k_syn, double kappa){
    Eigen::VectorXd v(2);
    v(0) = kruells94_beta(p.x(0), Xsh, a, b);
    v(1) = - p.x(1) * kruells94_dbetadx(p.x(0), Xsh, b) / 3 - k_syn * p.x(1) * p.x(1); 
    return v;
}

Eigen::MatrixXd kruells_shockaccel3_diffusion(const SpaceTimePoint& p, double kappa){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = sqrt(2 * kappa);
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = 0;
    return v;
}

BatchKruellsB1::BatchKruellsB1(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);

    double a1 = msa(params, "a1");
    double r = msa(params, "r");
    double gamma = msa(params, "gamma");
    double k_syn = msa(params, "k_syn");
    double a2 = a1 / (r * r);
    double beta_s = sqrt(4 * gamma);
    double kappa = gamma / (a1 * k_syn);

    std::cout << "a2 " << a2 << ", beta_s " << beta_s << ", kappa " << kappa << "\n";
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(beta_s, r);
    double b = b_from_shockparam(beta_s, r);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel3_drift_94, _1, msa(params, "Xsh"), a, b, k_syn, kappa);
    auto call_diffusion = std::bind(kruells_shockaccel3_diffusion, _1, kappa);
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}


BatchKruellsB1::~BatchKruellsB1(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

BatchKruellsB2::BatchKruellsB2(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    //Eigen::VectorXd xmin(2), xmax(2);
    //xmin << -L, 0;
    //xmax << L, 1000;
    //_slimit = new BreakpointSpatial(xmin, xmax);

    double a1 = msa(params, "a1");
    double r = msa(params, "r");
    double gamma = msa(params, "gamma");
    double k_syn = msa(params, "k_syn");
    double a2 = a1 / (r * r);
    double beta_s = sqrt(4 * gamma);
    double kappa = gamma / (a1 * k_syn);

    std::cout << "a2 " << a2 << ", beta_s " << beta_s << ", kappa " << kappa << "\n";
    
    // calculate a, b from shock max and compression ratio
    double a = a_from_shockparam(beta_s, r);
    double b = b_from_shockparam(beta_s, r);

    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_shockaccel3_drift_94, _1, msa(params, "Xsh"), a, b, k_syn, kappa);
    auto call_diffusion = std::bind(kruells_shockaccel3_diffusion, _1, kappa);
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new SemiImplicitWeakScheme2(call_drift, call_diffusion, call_timestep, _process);

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
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}


BatchKruellsB2::~BatchKruellsB2(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}

// *************** KRUELLS C1 *************
// 2nd order fermi (Cx)
//

Eigen::MatrixXd kruells_2ndorder_diffusion(const SpaceTimePoint& p, double kappa, double a2){
    Eigen::MatrixXd v(2, 2);
    v(0, 0) = std::sqrt(2 * kappa);
    v(0, 1) = 0;
    v(1, 0) = 0;
    v(1, 1) = std::sqrt(2 * a2) * p.x(1);
    return v;
}


Eigen::VectorXd kruells_2ndorder_drift(const SpaceTimePoint& p, double k_syn, double a2){
    Eigen::VectorXd v(2);
    v(0) = 0;//3 * a1;
    v(1) = p.x(1) * (4 * a2 - k_syn * p.x(1));
    return v;
}

BatchKruellsC1::BatchKruellsC1(std::map<std::string, double> params){
    // get a random generator
    _process = new WienerProcess(2);

    // time limit breakpoint
    _tlimit = new BreakpointTimelimit(msa(params, "Tmax"));

    // spatial breakpoint
    Eigen::VectorXd xmin(2), xmax(2);
    xmin << -msa(params, "Lx"), msa(params, "Lylower");
    xmax << msa(params, "Lx"), msa(params, "Lyupper");
    _slimit = new BreakpointSpatial(xmin, xmax);


    // callbacks
    // not sure if &function is better
    auto call_drift = std::bind(kruells_2ndorder_drift, _1, msa(params, "k_syn"), msa(params, "a2"));
    auto call_diffusion = std::bind(kruells_2ndorder_diffusion, _1, msa(params, "kappa"), msa(params, "a2"));
    auto call_timestep = std::bind(ts_const, _1, msa(params, "dt"));
    _scheme = new EulerScheme(call_drift, call_diffusion, call_timestep, _process);

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
    opt.breakpoints.push_back(_slimit);
    opt.scheme = _scheme;
    opt.tracked = false;

    // initialize
    initialize(starts, opt);
}


BatchKruellsC1::~BatchKruellsC1(){
    delete _scheme;
    delete _process;
    delete _tlimit;
    //delete _slimit;
    //delete _sintegrator;
}
