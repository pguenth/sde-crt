#include "pseudoparticlebatch.h"
PseudoParticleBatch::PseudoParticleBatch(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options) :
    _N(N), _callbacks(callbacks), _options(options), _start(start) {
        _construct();
}

PseudoParticleBatch::PseudoParticleBatch(int N, PseudoParticleCallbacks callbacks, double t0, Eigen::VectorXd x0, PseudoParticleOptions options) : _N(N), _callbacks(callbacks), _options(options) {
    _start = SpaceTimePoint(t0, x0);
    _construct();
}

PseudoParticleBatch::PseudoParticleBatch(int N, drift_t drift, diffusion_t diffusion, SpaceTimePoint start, PseudoParticleOptions options) : _N(N), _options(options), _start(start) {
    _callbacks = PseudoParticleCallbacks(drift, diffusion);
    _construct();
}

PseudoParticleBatch::PseudoParticleBatch(int N, drift_t drift, diffusion_t diffusion, double t0, Eigen::VectorXd x0, PseudoParticleOptions options) : _N(N), _options(options) {
    _start = SpaceTimePoint(t0, x0);
    _callbacks = PseudoParticleCallbacks(drift, diffusion);
    _construct();
}

PseudoParticleBatch::PseudoParticleBatch() {}

void PseudoParticleBatch::initialize(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options){
    _N = N;
    _callbacks = callbacks;
    _start = start;
    _options = options;
    _construct();
}



//PseudoParticleBatch::~PseudoParticleBatch();
//PseudoParticleBatch::PseudoParticleBatch(const PseudoParticleBatch&);
//PseudoParticleBatch::PseudoParticleBatch& operator= (const PseudoParticleBatch&);
//PseudoParticleBatch::PseudoParticleBatch(PseudoParticleBatch&&);
//PseudoParticleBatch::PseudoParticleBatch& operator= (PseudoParticleBatch&&);

//PseudoParticleBatch PseudoParticleBatch::operator+ (const PseudoParticleBatch&){
//    if (_
//}

const PseudoParticle& PseudoParticleBatch::operator[] (int index){
    if (index < 0 || index >= _N){
        throw std::invalid_argument("Index out of range");
    }

    return _particles[index];
}

void PseudoParticleBatch::_construct(){
    _particles = std::vector<PseudoParticle>(_N, PseudoParticle(_callbacks, _start, _options));
    _finished_count = 0;
    _initialized = true;
}

void PseudoParticleBatch::_check_init(){
    if (!_initialized) throw std::runtime_error("Using uninitialized PseudoParticleBatch");
}

bool PseudoParticleBatch::_run_one(int index){
    if (!_particles[index].finished()){
        _particles[index].run();
        return true;
    }else{
        return false;
    }
}

void PseudoParticleBatch::_one_step_all(){
    for (auto& p : _particles){
        if (!p.finished() && p.step()){
            _finished_count++;
        }
    }
}


// advances all unfinished particles for the given count of steps
//
// returns the number of finished particles
int PseudoParticleBatch::step_all(int steps){
    _check_init();

    while (steps-- > 0 && !finished()) _one_step_all();
    return finished_count();
}


// runs the given amount of particles, if -1 is given,
// all particles will be simulated
//
// returns the number of finished particles
int PseudoParticleBatch::run(int particle_count){
    _check_init();

    if (particle_count == -1) particle_count = _N;

    for (int i = 0; i < _N; i++){
        if (particle_count <= 0) break;
        if (_run_one(i)) particle_count--;
        //std::cout << "Running " << i << "\n";
    }

    return finished_count();
}

bool PseudoParticleBatch::finished(){
    _check_init();

    if (finished_count() == _N) return true;
    else return false;
}

int PseudoParticleBatch::finished_count(){
    return _finished_count;
}

int PseudoParticleBatch::unfinished_count(){
    return _N - _finished_count;
}


std::vector<PseudoParticle>& PseudoParticleBatch::particles(){
    _check_init();

    return _particles;
}

const std::vector<PseudoParticleState> PseudoParticleBatch::states(){
    _check_init();

    std::vector<PseudoParticleState> vec;
    
    for (auto& p : _particles)
        vec.push_back(p.state());

    return vec;
}

const PseudoParticleState& PseudoParticleBatch::state(int index){
    _check_init();

    return (*this)[index].state();
}

const PseudoParticle& PseudoParticleBatch::particle(int index){
    _check_init();

    return (*this)[index];
}

std::vector<double> PseudoParticleBatch::apply_integrator(TrajectoryIntegrator& integrator){
    std::vector<double> results;

    for (auto& p : _particles){
        if (p.finished()){
            results.push_back(integrator.integrate(p.state().get_trajectory()));
        }
    }

    return results;
}

