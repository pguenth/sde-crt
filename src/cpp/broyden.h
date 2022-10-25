#ifndef BROYDEN_H
#define BROYDEN_H

#include <iostream>
#include <Eigen/Core>

//typedef Eigen::VectorXd (*eigen_vecfun)(Eigen::VectorXd);
typedef std::function<Eigen::VectorXd(Eigen::VectorXd)> eigen_vecfun;

// calculate a column of the jacobi matrix, i.e. df/dx_j at x where f is
// a vector and the derivative then also.
Eigen::VectorXd jacobi_col(eigen_vecfun f, int j, Eigen::VectorXd x);

// calculate the jacobi matrix using the above function
Eigen::MatrixXd jacobian(eigen_vecfun f, Eigen::VectorXd x);

// this implements the sherman-morrison formula mentioned in https://en.wikipedia.org/wiki/Broyden%27s_method
Eigen::MatrixXd jacobi_inv(Eigen::MatrixXd j_inv_old, Eigen::VectorXd dx, Eigen::VectorXd df);

// https://en.wikipedia.org/wiki/Broyden%27s_method
Eigen::VectorXd broyden(eigen_vecfun f, Eigen::VectorXd x0, double tol = 1e-8, int maxiter = 200, bool *converged = nullptr);

#endif
