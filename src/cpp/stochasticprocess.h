#ifndef STOCHASTICPROCESS_H
#define STOCHASTICPROCESS_H

#include <Eigen/Core>

//template <class SeedType>
class StochasticProcess {
    protected:
        int _ndim;

    public:
        StochasticProcess(int ndim, void *seed = nullptr);
        virtual Eigen::VectorXd next(double timestep) = 0;
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
