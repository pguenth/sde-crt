#ifndef PSEUDOPARTICLEBATCH_H
#define PSEUDOPARTICLEBATCH_H

#include <vector>
#include "pseudoparticle.h"

class PseudoParticleBatch {
    private:
        int _N;
        std::vector<PseudoParticle> _particles;

        PseudoParticleOptions _options;
        PseudoParticleCallbacks _callbacks;
        SpaceTimePoint _start;

        int _finished_count;

        void _construct();
        void _one_step_all();
        bool _run_one(int index);

        bool _initialized = false;

        void _check_init();

    protected:
        void initialize(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options);

    public:
        PseudoParticleBatch();
        PseudoParticleBatch(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options);
        PseudoParticleBatch(int N, PseudoParticleCallbacks callbacks, double t0, Eigen::VectorXd x0, PseudoParticleOptions options);
        PseudoParticleBatch(int N, drift_t drift, diffusion_t diffusion, SpaceTimePoint start, PseudoParticleOptions options);
        PseudoParticleBatch(int N, drift_t drift, diffusion_t diffusion, double t0, Eigen::VectorXd x0, PseudoParticleOptions options);

        //~PseudoParticleBatch();
        //PseudoParticleBatch(const PseudoParticleBatch&);
        //PseudoParticleBatch& operator= (const PseudoParticleBatch&);
        //PseudoParticleBatch(PseudoParticleBatch&&);
        //PseudoParticleBatch& operator= (PseudoParticleBatch&&);

        //PseudoParticleBatch operator + (const PseudoParticleBatch&);
        const PseudoParticle& operator [] (int index);

        // advances all unfinished particles for the given count of steps
        //
        // returns the number of finished particles
        int step_all(int steps = 1);

        // runs the given amount of particles, if -1 is given,
        // all particles will be simulated
        //
        // returns the number of finished particles
        int run(int particle_count = -1);

        bool finished();
        int finished_count();
        int unfinished_count();


        std::vector<PseudoParticle>& particles();
        const std::vector<PseudoParticleState> states();
        const PseudoParticleState& state(int index);
        const PseudoParticle& particle(int index);

        // integrate all finished particles with the given integrator
        // returns a vector of the integration results
        std::vector<double> apply_integrator(TrajectoryIntegrator& integrator);
};




#endif
