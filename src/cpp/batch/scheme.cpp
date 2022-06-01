#include "scheme.h"

SDEScheme::SDEScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    _drift(drift), _diffusion(diffusion), _timestep(timestep), _process(process) {}

double SDEScheme::timestep_at(const SpaceTimePoint& p) const {
    return _timestep(p);
}

Eigen::VectorXd SDEScheme::next_random() const {
    return _process->next();
}

EulerScheme::EulerScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDESchemeCopyable(drift, diffusion, timestep, process) {}

SpaceTimePoint EulerScheme::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    Eigen::VectorXd x_new = p.x + ts * _drift(p) + _diffusion(p) * next_random() * sqrt(ts);   
    return SpaceTimePoint(p.t + ts, x_new);
}

//Scheme *EulerScheme::copy(std::vector<uint64_t> seeds) const {
//    StochasticProcess *new_p = _process->copy(seeds);
//    return new EulerScheme(_drift, _diffusion, _timestep, new_p);
//}

KPPCScheme::KPPCScheme(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDESchemeCopyable(drift, diffusion, timestep, process) {}

SpaceTimePoint KPPCScheme::propagate(const SpaceTimePoint& p) const {
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
    // in these steps, rnd is only the standard-normal but the sqrt(delta t) is in every diff_* term
    double dxdiff_corr = (rnd(0) * (diff_pls(0, 0) + diff_mns(0, 0) + 2 * diff_0(0, 0)) + (rnd(0) * rnd(0) - 1) * (diff_pls(0, 0) - diff_mns(0, 0))) / 4; // dx_diff bar (eq. 25)

    // is the initial time the right choice here for the drift term evaluation point
    double x_pred = p.x(0) + (_drift(SpaceTimePoint(p.t, xy_ces))(0) * ts + drift_0(0)) / 2 + dxdiff_corr; // xbar (eq. 23)
                                                                                                           //
    Eigen::VectorXd xy_pred_fake {{x_pred, 0.0}};
    // is the initial time the right choice here for the drift term evaluation point
    double x_c = p.x(0) + (_drift(SpaceTimePoint(p.t, xy_pred_fake))(0) * ts + drift_0(0)) / 2 + dxdiff_corr; //x_c (eq. 27)

    Eigen::VectorXd x_new {{x_c, xy_ces(1)}};
    return SpaceTimePoint(p.t + ts, x_new);
}

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

SemiImplicitWeakScheme2::SemiImplicitWeakScheme2(drift_t drift, diffusion_t diffusion, timestep_t timestep, StochasticProcess *process) :
    SDESchemeCopyable(drift, diffusion, timestep, process) {}

SpaceTimePoint SemiImplicitWeakScheme2::propagate(const SpaceTimePoint& p) const {
    double ts = timestep_at(p);
    bool has_converged;
    Eigen::VectorXd rnd = next_random();
    Eigen::VectorXd x_bar_solution = broyden([=](Eigen::VectorXd x_bar){return get_implicit(x_bar, p, ts, rnd);}, p.x, 1e-8, 2000, &has_converged);
    if (!has_converged){
    //    std::cout << "no convergence. random numbers are " << rnd(0) << " and " << rnd(1) << "\n";
    }else{
        //std::cout << "other random numbers are " << rnd(0) << " and " << rnd(1) << "\n";
    }


    double b = 0.0225;
    double xsh = 0.002;
    double alpha_max = - kruells94_dbetadx(0, xsh, b);
    double y_bar_analytic_min = p.x(1) / (1 - 0.001 * alpha_max / 2);
    
    Eigen::VectorXd x_nplus1 = 2 * x_bar_solution - p.x;

//    if (x_nplus1(1) <= 1){
//        // should be > 1 because the momentum should only increase 
//        // it could decrease if the timestep is too large (controllable) or if the C_ij term becomes too large (harder to control)
//
//        std::cout << "y_bar_analytic_min: " << y_bar_analytic_min << "; x_bar_solution: " << x_bar_solution(0) << "; y_bar_solution: " << x_bar_solution(1) << "\n";
//        
//        Eigen::VectorXd x_nplus1_euler = p.x + _drift(p) * ts + _diffusion(p) * rnd * sqrt(ts);
//        std::cout << "x(n+1): " << x_nplus1(0) << "; y(n+1): " << x_nplus1(1) << "\n";
//        std::cout << "Euler: x(n+1): " << x_nplus1_euler(0) << "; y(n+1): " << x_nplus1_euler(1) << "\n";
//        std::cout << "drift x: " << _drift(p)(0) << "; drift y: " << _drift(p)(1) << "\n";
//
//        if (x_bar_solution(1) < y_bar_analytic_min)
//            std::cout << "y_bar_solution too small!\n";
//
//        if (x_bar_solution(1) < p.x(1))
//            std::cout << "y_bar < y: " << x_bar_solution(1) << " < " << p.x(1) << "\n";
//    }

    return SpaceTimePoint(p.t + ts, x_nplus1);
}

std::vector<Eigen::MatrixXd> SemiImplicitWeakScheme2::get_B_diff(const SpaceTimePoint& p, double delta = 0.00001) const {
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

Eigen::VectorXd SemiImplicitWeakScheme2::get_C(const SpaceTimePoint& p_bar) const {
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

Eigen::VectorXd SemiImplicitWeakScheme2::get_implicit(Eigen::VectorXd x_bar, const SpaceTimePoint& p, double timestep, Eigen::VectorXd rnd) const {
    SpaceTimePoint p_bar(p.t + timestep / 2, x_bar);
    Eigen::VectorXd C = get_C(p_bar);
    Eigen::VectorXd diff =  sqrt(timestep) * _diffusion(p_bar) * rnd;
    Eigen::VectorXd drift = _drift(p_bar) * timestep;
    //if (not C(1) == 0)
    //    std::cout << "C is " << C(0) << ", " << C(1) << "\n";
    //if (not diff(1) == 0)
    //    std::cout << "diff is " << diff(0) << ", " << diff(1) << "\n";
    //if (drift(1) < 0)
    //    std::cout << "drift is " << drift(0) << ", " << drift(1) << "\n";

    //std::cout << "Test scheme2\n";
    return 2 * p.x - 2 * x_bar + drift - C / 2 * timestep + diff;
}
