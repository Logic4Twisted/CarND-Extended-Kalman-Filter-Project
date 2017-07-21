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
  MatrixXd S = H_*P_*H_.transpose() + R_;
  MatrixXd K = P_*H_.transpose()*S.inverse();
  x_ = x_ + K*y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  const double pi = 3.1415926535897;
  float r = z(0);
  float phi = z(1);
  float v = z(2);
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  std::cout << ">" << px << "," << py << "<" << std::endl;
  VectorXd h_x = VectorXd(3);
  if (px*px + py*py < 1.0e-5) {
    return;
  }
  else if (fabs(px) < 1.0e-5) {
    if (py > 0) {
      h_x << py, pi/2, vy;
    }
    else {
      h_x << py, -pi/2, vy;
    }
  }
  else {
    float fi = atan2(py, px); // ? +-pi
    float s = sqrt(px*px + py*py);
    h_x << s, fi, (px*vx+py*vy)/s;
  }
  std::cout << "h_x = " << h_x << std::endl;
  VectorXd y = z - h_x;
  Tools tools = Tools();
  VectorXd x_predicted(4);
  x_predicted << r*cos(phi), r*sin(phi), v*cos(phi), r*sin(phi);
  MatrixXd Hj = tools.CalculateJacobian(x_predicted);
  std::cout << "Hj = " << Hj << std::endl;
  MatrixXd S = Hj*P_*Hj.transpose() + R_;
  MatrixXd K = P_*Hj.transpose()*S.inverse();
  x_ = x_ + K*y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K*Hj)*P_;
}
