#include "broyden.h"

// calculate a column of the jacobi matrix, i.e. df/dx_j at x where f is
// a vector and the derivative then also.
Eigen::VectorXd jacobi_col(eigen_vecfun f, int j, Eigen::VectorXd x){
    double h = 0.00001;
    Eigen::VectorXd x_0 = x;
    Eigen::VectorXd x_1 = x;
    x_0(j) -= h / 2;   
    x_1(j) += h / 2;
    Eigen::VectorXd f_0 = f(x_0);
    Eigen::VectorXd f_1 = f(x_1);
    return (f_1 - f_0) / h;
}

// calculate the jacobi matrix using the above function
Eigen::MatrixXd jacobian(eigen_vecfun f, Eigen::VectorXd x){
    Eigen::MatrixXd jac(x.rows(), x.rows());
    Eigen::VectorXd col(x.rows());
    for (int j = 0; j < x.rows(); j++){
        col = jacobi_col(f, j, x);
        for (int i = 0; i < x.rows(); i++){
            jac(i, j) = col(i);
        }
    }
    return jac;
}

// this implements the sherman-morrison formula mentioned in https://en.wikipedia.org/wiki/Broyden%27s_method
Eigen::MatrixXd jacobi_inv(Eigen::MatrixXd j_inv_old, Eigen::VectorXd dx, Eigen::VectorXd df){
    Eigen::MatrixXd pre_matrix_num = (dx - j_inv_old * df) * dx.transpose();
    double pre_matrix_denom = dx.transpose() * j_inv_old * df;
    if (pre_matrix_denom == 0){
        return j_inv_old;
    }else{
        Eigen::MatrixXd increment = (pre_matrix_num * j_inv_old) / pre_matrix_denom;
        return j_inv_old + increment;
    }
}

// https://en.wikipedia.org/wiki/Broyden%27s_method
Eigen::VectorXd broyden(eigen_vecfun f, Eigen::VectorXd x0, double tol /*= 1e-8*/, int maxiter /*= 200*/, bool *converged /*= nullptr*/){
    Eigen::VectorXd x_new(x0.rows());
    Eigen::VectorXd x_old(x0.rows());
    Eigen::VectorXd dx(x0.rows());
    Eigen::VectorXd f_new(x0.rows());
    Eigen::VectorXd f_old(x0.rows());
    Eigen::VectorXd df(x0.rows());
    Eigen::MatrixXd jacobi;
    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", " | ", "", "", "(", ")");

    x_new = x0;
    x_old = x0;
    //std::cout << x0 << x_new << "\n";
    for (int i = 0; i < x0.rows(); i++) f_new(i) = 0;
    f_old = f_new;
    jacobi = jacobian(f, x0);

    int iter = 0;
    double residual = 1;
    while (iter++ < maxiter and tol < residual){
        // going from n -> n+1
        f_old = f_new; // f_n-1 
        f_new = f(x_new); // f_n
        df = f_new - f_old;
        dx = x_new - x_old;
        jacobi = jacobi_inv(jacobi, dx, df); // J_n
        //std::cout << "iter " << iter << ": x_n-1 = " << x_old.format(fmt) << "; x_n = " << x_new.format(fmt) << "; f_n-1 =  " << f_old.format(fmt) << "; f_n = "  << f_new.format(fmt) << "; Jacobian: " << jacobi.format(fmt) << "; Residual: " << residual << "\n";

        x_old = x_new; // x_n
        x_new = x_new - jacobi * f_new; // x_n+1
        
        residual = 0;
        for (int i = 0; i < x0.rows(); i++) residual += f_new(i) * f_new(i);
    }

    if (converged != nullptr) *converged = (tol > residual);
    //if (tol < residual)
    //    std::cout << "broyden does not converge and used " << iter << " iterations and the final tolerance is " << residual << "\n";

    return x_new;
}

