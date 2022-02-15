#ifndef STOCHASTICPROCESS_H
#define STOCHASTICPROCESS_H

#include <Eigen/Core>

//template <class SeedType>
class StochasticProcess {
    protected:
        int _ndim;

    public:
        StochasticProcess(int ndim, std::vector<uint64_t> seeds = {});
        virtual ~StochasticProcess() = default;
        virtual Eigen::VectorXd next() = 0;
        virtual StochasticProcess *copy(std::vector<uint64_t> seeds) = 0;
};

//class DummyProcess : public StochasticProcess<int> {
//    protected:
//        int _ndim;
//
//    public:
//        DummyProcess(int ndim, int seed = 0);
//        Eigen::VectorXd next(double timestep);
//};



#endif
