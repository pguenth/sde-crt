#include "pseudoparticlebatch.h"
PseudoParticleBatch::PseudoParticleBatch(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options){
    initialize(N, callbacks, start, options);
}

PseudoParticleBatch::PseudoParticleBatch(int N, PseudoParticleCallbacks callbacks, double t0, Eigen::VectorXd x0, PseudoParticleOptions options){
    SpaceTimePoint start(t0, x0);
    initialize(N, callbacks, start, options);
}

PseudoParticleBatch::PseudoParticleBatch(int N, drift_t drift, diffusion_t diffusion, SpaceTimePoint start, PseudoParticleOptions options){
    PseudoParticleCallbacks callbacks(drift, diffusion);
    initialize(N, callbacks, start, options);
}

PseudoParticleBatch::PseudoParticleBatch(int N, drift_t drift, diffusion_t diffusion, double t0, Eigen::VectorXd x0, PseudoParticleOptions options){
    SpaceTimePoint start(t0, x0);
    PseudoParticleCallbacks callbacks(drift, diffusion);
    initialize(N, callbacks, start, options);
}

PseudoParticleBatch::PseudoParticleBatch() {}

void PseudoParticleBatch::initialize(int N, PseudoParticleCallbacks callbacks, SpaceTimePoint start, PseudoParticleOptions options){
    std::vector<SpaceTimePoint> starts(N, start);
    initialize(callbacks, starts, options);
}

void PseudoParticleBatch::initialize(PseudoParticleCallbacks callbacks, std::vector<SpaceTimePoint> starts, PseudoParticleOptions options){
    _construct(callbacks, starts, options);
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
    if (index < 0 || index >= count()){
        throw std::invalid_argument("Index out of range");
    }

    return _particles[index];
}

void PseudoParticleBatch::_construct(PseudoParticleCallbacks callbacks, std::vector<SpaceTimePoint> starts, PseudoParticleOptions options){
    _callbacks = callbacks;
    _options = options;
    _starts = starts;

    _finished_count = 0;
    _initialized = true;

    _particles = std::vector<PseudoParticle>();

    std::seed_seq seq{};
    std::vector<std::uint32_t> seeds(_starts.size());
    seq.generate(seeds.begin(), seeds.end());
    auto seed = seeds.begin();

    for (auto &s: _starts){
        _options.seed = *seed;
        seed = std::next(seed);
        _particles.push_back(PseudoParticle(_callbacks, s, _options));
    }
}

void PseudoParticleBatch::_check_init(){
    if (!_initialized) throw std::runtime_error("Using uninitialized PseudoParticleBatch");
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

/*
int PseudoParticleBatch::run(int particle_count){
    _check_init();

    if (particle_count == -1) particle_count = count();

    boost::asio::thread_pool pool(NTHREADS);

    for (int i = 0; i < particle_count; i++){
        boost::asio::post(pool, [this, i](){ _particles[i].run(); });
    }

    pool.join();

    return finished_count();
}

*
 * manual threading 
 */


struct ThreadArgs {
    PseudoParticleBatch *batch;
    int mod_base;
    int mod_res;
};

void *run_wrapper(void *arg){
    struct ThreadArgs args = *(static_cast<ThreadArgs *>(arg));
    args.batch->run_mod(args.mod_base, args.mod_res);
    return nullptr;
}

void PseudoParticleBatch::run_mod(int mod_base, int mod_res){
    int run_count = 0;
    for (int i = 0; i < count(); i++){
        //std::cout << "thread " << mod_res << " of " << mod_base << " start\n";
        if (i % mod_base == mod_res){
            //std::cout << "thread " << mod_res << " of " << mod_base << " running particle " << i << "\n";
            run_count++;
            _particles[i].run();
        }
    }
    std::cout << "thread " << mod_res << " of " << mod_base << " runned " << run_count << " particles\n";
}

int PseudoParticleBatch::run(int particle_count){
    int nthreads = 1;

    std::vector<pthread_t> threads(nthreads);
    std::vector<struct ThreadArgs> thread_args(nthreads);
    for (int i = 0; i < nthreads; i++){
        //auto callback = [this, nthreads, i]() { run_mod(nthreads, i); };
        thread_args[i].batch = this;
        thread_args[i].mod_base = nthreads;
        thread_args[i].mod_res = i;
        pthread_create(&(threads[i]), nullptr, &run_wrapper, &(thread_args[i]));
    }

    void *ret;
    for (auto& thread : threads){
        pthread_join(thread, &ret);
    }
    
    return 0;
}

bool PseudoParticleBatch::finished(){
    _check_init();

    if (finished_count() == count()) return true;
    else return false;
}

int PseudoParticleBatch::finished_count(){
    return _finished_count;
}

int PseudoParticleBatch::unfinished_count(){
    return count() - _finished_count;
}

int PseudoParticleBatch::count(){
    return _starts.size();
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


std::vector<std::vector<double>> PseudoParticleBatch::get_integrator_values(){
    std::vector<std::vector<double>> ret;

    for (auto& p : _particles){
        ret.push_back(std::vector<double>());
        auto& pvec = ret.back();
        for (auto& integrator : p.integrators()){
            pvec.push_back(integrator->value());
        }
    }

    return ret;
}
