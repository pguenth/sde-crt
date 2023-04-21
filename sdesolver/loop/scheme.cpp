#include "scheme.h"

scheme_t scheme_registry_lookup(const std::string& name){
    std::map<std::string, scheme_t> s_map;

    s_map["euler"] = scheme_euler;
    s_map["kppc"] = scheme_kppc;
    s_map["semiimplicit_weak"] = scheme_semiimplicit_weak;
    
    return s_map[name];
}

double scheme_euler(Eigen::VectorXd& x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion) {
    int ndim = x.rows();
    Eigen::VectorXd drift_buf(ndim);
    Eigen::MatrixXd diffusion_buf(ndim, ndim);

    drift(drift_buf.data(), t, x.data());
    diffusion(diffusion_buf.data(), t, x.data());

    x_out = x + timestep * drift_buf + diffusion_buf * rndvec * sqrt(timestep);
    return t + timestep;
}

double scheme_kppc(Eigen::VectorXd& x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion) {
    double ts = timestep;
    Eigen::VectorXd rnd = rndvec;
    double sts = sqrt(ts);

    int ndim = x.rows();
    Eigen::VectorXd drift_buf(ndim);
    Eigen::MatrixXd diffusion_buf(ndim, ndim);
    drift(drift_buf.data(), t, x.data());
    diffusion(diffusion_buf.data(), t, x.data());



    Eigen::VectorXd unit2d(2);
    unit2d << 1.0, 1.0;
    Eigen::VectorXd drift_0 = drift_buf * ts;
    Eigen::MatrixXd diff_0 = diffusion_buf * sts;
    Eigen::VectorXd xy_ces = x + drift_0 + diff_0 * rnd; // \tilde x (eq. 21)
    Eigen::VectorXd xy_pls = x + drift_0 + diff_0 * unit2d; // x+ (eq. 22)
    Eigen::VectorXd xy_mns = x + drift_0 - diff_0 * unit2d; // x- (eq. 22)

    Eigen::MatrixXd diffusion_buf2(ndim, ndim);
    Eigen::MatrixXd diffusion_buf3(ndim, ndim);

    diffusion(diffusion_buf2.data(), t, xy_pls.data());
    Eigen::MatrixXd diff_pls = diffusion_buf2 * sts;

    diffusion(diffusion_buf3.data(), t, xy_mns.data());
    Eigen::MatrixXd diff_mns = diffusion_buf3 * sts;
    // in these steps, rnd is only the standard-normal but the sqrt(delta t) is in every diff_* term
    double dxdiff_corr = (rnd(0) * (diff_pls(0, 0) + diff_mns(0, 0) + 2 * diff_0(0, 0)) + (rnd(0) * rnd(0) - 1) * (diff_pls(0, 0) - diff_mns(0, 0))) / 4; // dx_diff bar (eq. 25)

    // is the initial time the right choice here for the drift term evaluation point
    Eigen::VectorXd drift_buf2(ndim);
    drift(drift_buf2.data(), t, xy_ces.data());
    double x_pred = x(0) + (drift_buf2(0) * ts + drift_0(0)) / 2 + dxdiff_corr; // xbar (eq. 23)
                                                                                                           //
    Eigen::VectorXd xy_pred_fake(2);
    xy_pred_fake << x_pred, 0.0;
    // is the initial time the right choice here for the drift term evaluation point
    Eigen::VectorXd drift_buf3(ndim);
    drift(drift_buf3.data(), t, xy_pred_fake.data());
    double x_c = x(0) + (drift_buf3(0) * ts + drift_0(0)) / 2 + dxdiff_corr; //x_c (eq. 27)

    x_out(0) = x_c;
    x_out(1) = xy_ces(1);
    return t + ts;
}

//
// Semi-Implicit
//

std::vector<Eigen::MatrixXd> semiimplicit_weak_get_B_diff(double t, const Eigen::VectorXd& x, coeff_call_t diffusion, double delta = 0.00001) {
    int ndim = x.rows();
    std::vector<Eigen::MatrixXd> r;
    Eigen::VectorXd ppls;
    Eigen::VectorXd pmns;
    Eigen::MatrixXd diff_ppls(ndim, ndim);
    Eigen::MatrixXd diff_pmns(ndim, ndim);
    //std::cout << "d 12\n";
    for (int k = 0; k < ndim; k++){
        ppls = x;
        pmns = x;
        ppls(k) += delta;
        pmns(k) -= delta;
        diffusion(diff_ppls.data(), t, ppls.data());
        diffusion(diff_pmns.data(), t, pmns.data());
        r.push_back((diff_ppls - diff_pmns) / (2 * delta));
    }
    //std::cout << "d 13\n";

    return r;
}

Eigen::VectorXd semiimplicit_weak_get_C(double t_bar, const Eigen::VectorXd& x_bar, coeff_call_t diffusion) {
    int ndim = x_bar.rows();
    //std::cout << "d 8\n";
    std::vector<Eigen::MatrixXd> difftensor = semiimplicit_weak_get_B_diff(t_bar, x_bar, diffusion);
    //std::cout << "d 9\n";
    Eigen::MatrixXd diffmatrix(ndim, ndim);
    diffusion(diffmatrix.data(), t_bar, x_bar.data());
    //std::cout << "d 10\n";
    Eigen::VectorXd r(ndim);
    for (int i = 0; i < ndim; i++){
        r(i) = 0;
        for (int j = 0; j < ndim; j++){
            for (int k = 0; k < ndim; k++){
                r(i) += diffmatrix(k, j) * difftensor.at(k)(i, j);
            }
        }
    }
    //std::cout << "d 11\n";

    return r;
}

Eigen::VectorXd semiimplicit_weak_get_implicit(Eigen::VectorXd x_bar, double t, const Eigen::VectorXd& x, double timestep, Eigen::VectorXd rnd, coeff_call_t drift_call, coeff_call_t diffusion_call) {
    int ndim = x.rows();
    double t_bar = t + timestep / 2;
    //std::cout << "d 3\n";
    Eigen::VectorXd C = semiimplicit_weak_get_C(t_bar, x_bar, diffusion_call);

    Eigen::MatrixXd diff_bar(ndim, ndim);
    diffusion_call(diff_bar.data(), t_bar, x_bar.data());
    //std::cout << "d 4\n";
    Eigen::VectorXd drift_bar(ndim);
    drift_call(drift_bar.data(), t_bar, x_bar.data());
    //std::cout << "d 5\n";

    Eigen::VectorXd diff = sqrt(timestep) * diff_bar * rnd;
    //std::cout << "d 6\n";
    Eigen::VectorXd drift = drift_bar * timestep;
    //std::cout << "d 7\n";
    
    return 2 * x - 2 * x_bar + drift - C / 2 * timestep + diff;
}

double scheme_semiimplicit_weak(Eigen::VectorXd& x_out, double t, const Eigen::Map<Eigen::VectorXd> &x, const Eigen::VectorXd& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion) {
    bool has_converged;
    //std::cout << "d 1\n";
    Eigen::VectorXd x_bar_solution = broyden([=](Eigen::VectorXd x_bar){return semiimplicit_weak_get_implicit(x_bar, t, x, timestep, rndvec, drift, diffusion);}, x, 1e-8, 2000, &has_converged);
    //std::cout << "d 2\n";
    Eigen::VectorXd result = 2 * x_bar_solution - x;
    //std::cout << "d 3\n";
    //std::cout << result(0) << ", " << result(1) << "\n";
    //std::cout << "d 4\n";
    x_out = result;
    //std::cout << "d 5\n";
    return t + timestep;
    //return x_out;
}


/*
ImplicitEulerScheme::ImplicitEulerScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) : 
    SDESchemeCopyable(drift, diffusion, timestep, process) {}
    
SpaceTimePoint ImplicitEulerScheme::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    Eigen::VectorXd r = next_random();
    //auto solvf = std::bind(ImplicitEulerScheme::get_implicit, std::placeholders::_1, p, ts);
    Eigen::VectorXd solution = broyden([=](Eigen::VectorXd x){return get_implicit(x, p, ts, r);}, p.x, 1e-8, 200);
    return SpaceTimePoint(p.t + ts, solution);
}

Eigen::VectorXd ImplicitEulerScheme::get_implicit(Eigen::VectorXd x_new, const SpaceTimePoint& p, double timestep, Eigen::VectorXd rnd) const {
    SpaceTimePoint p_new(p.t + timestep, x_new);
    return x_new - (p.x + timestep * _drift(p_new) + _diffusion(p) * rnd * sqrt(timestep));
}

SecondOrderScheme::SecondOrderScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDESchemeCopyable(drift, diffusion, timestep, process) {}

SpaceTimePoint SecondOrderScheme::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    Eigen::VectorXd rnd = next_random();
    double sts = sqrt(ts);
    Eigen::VectorXd unit2d {{1.0, 1.0}};
    Eigen::VectorXd drift_0 = _drift(p) * ts;
    Eigen::MatrixXd diff_0 = _diffusion(p) * sts;
    Eigen::VectorXd xy_ces = p.x + drift_0 + diff_0 * rnd; // \tilde x (eq. 21)
    Eigen::VectorXd xy_pls = p.x + drift_0 + diff_0 * unit2d; // x+ (eq. 22)
    Eigen::VectorXd xy_mns = p.x + drift_0 - diff_0 * unit2d; // x- (eq. 22)
    Eigen::MatrixXd diff_pls = _diffusion(SpaceTimePoint(p.t, xy_pls)) * sts;
    Eigen::MatrixXd diff_mns = _diffusion(SpaceTimePoint(p.t, xy_mns)) * sts;

    // is the initial time the right choice here for the drift term evaluation point
    double drift_ces_x = ts * _drift(SpaceTimePoint(p.t, xy_ces))(0);

    // it would be correct to write: (Kloeden/Platen ch. 15.1. eq. 1.3)
    // double factor = pow(rnd(0), 2) - 1;
    // but I leave it for legacy
    double factor = (pow(rnd(0), 2) - ts) / ts;
    double x_new = p.x(0) 
                 + (drift_ces_x + drift_0(0)) / 2 
                 + rnd(0) * (diff_pls(0, 0) + diff_mns(0, 0) + 2 * diff_0(0, 0)) / 4 
                 + factor * (diff_pls(0, 0) - diff_mns(0, 0)) / 4;

    Eigen::VectorXd xy_new {{x_new, xy_ces(1)}};
    return SpaceTimePoint(p.t + ts, xy_new);
}

SecondOrderScheme2::SecondOrderScheme2(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDESchemeCopyable(drift, diffusion, timestep, process) {}

SpaceTimePoint SecondOrderScheme2::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    Eigen::VectorXd rnd = next_random();
    double sts = sqrt(ts);
    Eigen::VectorXd unit2d {{1.0, 1.0}};
    Eigen::VectorXd drift_0 = _drift(p) * ts;
    Eigen::MatrixXd diff_0 = _diffusion(p) * sts;
    Eigen::VectorXd xy_ces = p.x + drift_0 + diff_0 * rnd; // \tilde x (eq. 21)
    Eigen::VectorXd xy_pls = p.x + drift_0 + diff_0 * unit2d; // x+ (eq. 22)
    Eigen::VectorXd xy_mns = p.x + drift_0 - diff_0 * unit2d; // x- (eq. 22)
    Eigen::MatrixXd diff_pls = _diffusion(SpaceTimePoint(p.t, xy_pls)) * sts;
    Eigen::MatrixXd diff_mns = _diffusion(SpaceTimePoint(p.t, xy_mns)) * sts;

    Eigen::VectorXd drift_ces_x = ts * _drift(SpaceTimePoint(p.t, xy_ces));

    // see SecondOrderScheme 2 for the error in this equation
    Eigen::VectorXd rndsq {{pow(rnd(0), 2) - ts, pow(rnd(1), 2) - ts}};
    Eigen::VectorXd factor = rndsq / sts;

    Eigen::VectorXd xy_new = p.x + (drift_ces_x + drift_0) / 2 + (diff_pls + diff_mns + 2 * diff_0) * rnd / 4 + (diff_pls(0, 0) - diff_mns(0, 0)) * factor / 4;

    return SpaceTimePoint(p.t + ts, xy_new);
}

SemiImplicitWeakScheme::SemiImplicitWeakScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDESchemeCopyable(drift, diffusion, timestep, process) {}

inline double kruells94_dbetadx(double x, double Xsh, double b){
    double csh = std::cosh(x / Xsh);
    if (csh == HUGE_VAL){
        return 0;
    }else{
        return -b / (Xsh * std::pow(csh, 2));
    }
}

SpaceTimePoint SemiImplicitWeakScheme::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    Eigen::VectorXd rnd = next_random();
    Eigen::VectorXd x_bar_solution = broyden([=](Eigen::VectorXd x_bar){return get_implicit(x_bar, p, ts, rnd);}, p.x, 1e-8, 200);
    
    return SpaceTimePoint(p.t + ts, 2 * x_bar_solution - p.x);
}

std::vector<Eigen::MatrixXd> SemiImplicitWeakScheme::get_B_diff(const SpaceTimePoint& p, double delta = 0.00001) const {
    int ndim = p.x.rows();
    std::vector<Eigen::MatrixXd> r;
    SpaceTimePoint ppls;
    SpaceTimePoint pmns;
    for (int k = 0; k < ndim; k++){
        ppls = p;
        pmns = p;
        ppls.x(k) += delta;
        pmns.x(k) -= delta;
        r.push_back((_diffusion(ppls) - _diffusion(pmns)) / (2 * delta));
    }

    return r;
}

Eigen::VectorXd SemiImplicitWeakScheme::get_C(const SpaceTimePoint& p_bar) const {
    int ndim = p_bar.x.rows();
    std::vector<Eigen::MatrixXd> difftensor = get_B_diff(p_bar);
    Eigen::MatrixXd diffmatrix = _diffusion(p_bar);
    Eigen::VectorXd r(ndim);
    for (int i = 0; i < ndim; i++){
        r(i) = 0;
        for (int j = 0; j < ndim; j++){
            for (int k = 0; k < ndim; k++){
                r(i) += diffmatrix(k, j) * difftensor.at(k)(i, j);
            }
        }
    }

    return r;
}


Eigen::VectorXd SemiImplicitWeakScheme::get_implicit(Eigen::VectorXd x_bar, const SpaceTimePoint& p, double timestep, Eigen::VectorXd rnd) const {
    SpaceTimePoint p_bar(p.t + timestep / 2, x_bar);
    return 2 * p.x - 2 * x_bar + _drift(p_bar) - get_C(p_bar) / 2 + _diffusion(p_bar) * rnd;
}
*/

