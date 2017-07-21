#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  if (estimations.size() == 0) return rmse;

  for (int i = 0; i < estimations.size(); i++) {
    VectorXd d = estimations[i].array()-ground_truth[i].array();
    d = d.array() * d.array();
    rmse = rmse + d;
  }
  rmse /= estimations.size();
  rmse = sqrt(rmse.array());
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj = MatrixXd(3,4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float sqr = px*px + py*py;
  if (sqr <= 1.0e-5) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }
  float s1 = sqrt(sqr);
  float s2 = s1*sqr;

  Hj << (px/s1), (py/s1), 0, 0,
      (-py/sqr), (px/sqr), 0, 0,
      py*(vx*py-vy*px)/s2, px*(vy*px-vx*py)/s2, px/s1, py/s1;

  return Hj;
}
