#ifndef PSEUDOPARTICLEBATCH_H
#define PSEUDOPARTICLEBATCH_H

#include <vector>
#include <thread>
#include "pseudoparticle.h"
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#define NTHREADS 8

class PseudoParticleBatch {
    private:
        std::vector<SpaceTimePoint> _starts;
        std::vector<PseudoParticle> _particles;

        PseudoParticleOptions _options;
        PseudoParticleCallbacks _callbacks;

        int _finished_count;

        /**
         * construct creates all particles and stores/initializes the member variables
         */
        void _construct(PseudoParticleCallbacks callbacks, std::vector<SpaceTimePoint> starts, PseudoParticleOptions options);

        void _one_step_all();
        bool _run_one(int index);

        bool _initialized = false;

        void _check_init();

    protected:
        /**
         * Fill the member variables and initialize the particles
         *
         * initialize() is called by constructors of this class or its children
         * to fill instances with sensible values from different sets of
         * parameters.
         *
         * @param N Number of pseudo particles
         * @param callbacks Callbacks for the pseudo particles
         * @param start Starting point of all pseudo particles
         * @param options
         */
        void initialize(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options);

        /**
         * @see initialize()
         * @param callbacks Callbacks for the pseudo particles
         * @param starts Vector of starting points for the pseudo particles. For every given element, one pseudo particle is created
         * @param options
         */
        void initialize(PseudoParticleCallbacks callbacks, std::vector<SpaceTimePoint> starts, PseudoParticleOptions options);

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

        /**
         * advances all unfinished particles for the given count of steps
         *
         * @param steps how many steps to advance all particles
         * @returns the number of finished particles
         */
        int step_all(int steps = 1);

        /**
         * runs the given amount of particles, if -1 is given,
         * all particles will be simulated 
         *
         * @param particle_count the number of particles to run
         * @return the number of finished particles
         */
        int run(int particle_count = -1);
        //int run_some(int particle_count = -1);
        void run_mod(int mod_base, int mod_res);

        bool finished();
        int finished_count();
        int unfinished_count();
        int count();


        std::vector<PseudoParticle>& particles();
        const std::vector<PseudoParticleState> states();
        const PseudoParticleState& state(int index);
        const PseudoParticle& particle(int index);

        /**
         * integrate all finished particles with the given integrator
         *
         * @param integrator Use this integrator to integrate over all trajectories
         * @return Integration results for every particle
         */
        std::vector<double> apply_integrator(TrajectoryIntegrator& integrator);

        /**
         * returns the value of all live integrators
         *
         * @return std::vector containing N std::vector, each containing M double values. N is the number of pseudo particles, M is the number if live integrators attached.
         */
        std::vector<std::vector<double>> get_integrator_values();
};




#endif
