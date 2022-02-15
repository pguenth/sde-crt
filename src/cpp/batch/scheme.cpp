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
    double dxdiff_corr = (rnd(0) * (diff_pls(0, 0) + diff_mns(0, 0) + 2 * diff_0(0, 0)) + (rnd(0) * rnd(0) - 1) * (diff_pls(0, 0) - diff_mns(0, 0))) / 4; // dx_diff bar (eq. 25)
    double x_pred = p.x(0) + (_drift(SpaceTimePoint(p.t, xy_ces))(0) * ts + drift_0(0)) / 2 + dxdiff_corr; // xbar (eq. 23)
    Eigen::VectorXd xy_pred_fake {{x_pred, 0.0}};
    double x_c = p.x(0) + (_drift(SpaceTimePoint(p.t, xy_pred_fake))(0) * ts + drift_0(0)) / 2 + dxdiff_corr; //x_c (eq. 27)

    Eigen::VectorXd x_new {{x_c, xy_ces(1)}};
    return SpaceTimePoint(p.t + ts, x_new);
}
