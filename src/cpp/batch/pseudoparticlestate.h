#ifndef PSEUDOPARTICLESTATE_H
#define PSEUDOPARTICLESTATE_H

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <algorithm>
#include "breakpointstate.h"
#include <sstream>

class PseudoParticle;
enum class BreakpointState;

typedef struct SpaceTimePoint {
    double t;
    Eigen::VectorXd x;

    SpaceTimePoint();
    SpaceTimePoint(double t, const Eigen::VectorXd& x);
    operator std::string() const;
} SpaceTimePoint;

typedef std::vector<SpaceTimePoint> trajectory_t;

class PseudoParticleState {
    private:
        const PseudoParticle *_particle;

        trajectory_t _trajectory;

        bool _finished;
        bool _tracked;
        BreakpointState _breakpoint_state;

    protected:
        void _pre_allocate(int max_steps);

    public:
        // constructors
        PseudoParticleState();
        PseudoParticleState(const PseudoParticle *particle, const SpaceTimePoint& p0, int max_steps = 0, bool tracked = true);
        PseudoParticleState(const PseudoParticle *particle, double t0, const Eigen::VectorXd& x0, int max_steps = 0, bool tracked = true);
        //~PseudoParticleState();
        

        // trajectory
        const Eigen::VectorXd& get_x() const;
        double get_t() const;
        const SpaceTimePoint& get_p() const;
        const trajectory_t& get_trajectory() const;
        bool tracked() const;

        // end state
        bool finished() const;
        const PseudoParticle *get_particle() const;
        BreakpointState get_breakpoint_state() const;

        // operators
        operator std::string() const; //string repr.
        const SpaceTimePoint& operator ()() const; //returns current space-time point

        void update(const SpaceTimePoint& p);
        void update(double t, const Eigen::VectorXd& x);
        void finish(BreakpointState breakpoint_state = BreakpointState::UNDEFINED);
};

#endif
