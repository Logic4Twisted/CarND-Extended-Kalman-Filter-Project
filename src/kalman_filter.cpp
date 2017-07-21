#include "kalman_filter.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_*P_*H_t + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd K = P_*H_t*S_i;
  x_ = x_ + K*y;
  int size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  P_ = (I - K*H_)*P_;
}

VectorXd KalmanFilter::getPolar() {
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  VectorXd h_x = VectorXd(3);
  float eps = 1.0e-5;
  if (fabs(px) < eps & fabs(px) < eps) {
    px = eps;
    py = eps;
  } else if (fabs(px) < 1.0e-5) {
    px = eps;
  }

  float fi = atan2f(py, px);
  float s = sqrtf(powf(px, 2) + powf(py, 2));
  h_x << s, fi, (px * vx + py * vy) / s;
  return h_x;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  VectorXd h_x = getPolar();

  std::cout << "h_x = " << h_x << std::endl;
  VectorXd y = z - h_x;
  y[1] -= (2 * M_PI) * floor((y[1] + M_PI) / (2 * M_PI));

  std::cout << "Hj = " << H_ << std::endl;
  MatrixXd Hj_t = H_.transpose();
  MatrixXd S = H_*P_*Hj_t + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd K = P_*Hj_t*S_i;
  x_ = x_ + K*y;

  int size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  P_ = (I - K*H_)*P_;
}
