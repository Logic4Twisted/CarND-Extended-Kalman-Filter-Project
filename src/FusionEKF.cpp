#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //mesurement function
  H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  ekf_ = KalmanFilter();

  tools = Tools();
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float r = measurement_pack.raw_measurements_[0];
      float fi = measurement_pack.raw_measurements_[1];
      std::cout << "Angle = " << fi << std::endl;
      float b = measurement_pack.raw_measurements_[2];
      ekf_.x_ << r*cos(fi), r*sin(fi), b*cos(fi), b*cos(fi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    std::cout << ekf_.x_ << std::endl;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  float dt = ((float)(measurement_pack.timestamp_ - previous_timestamp_))/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1.0, 0, dt, 0,
             0, 1.0, 0, dt,
             0, 0, 1.0, 0,
             0, 0, 0, 1.0;

  ekf_.Q_ = MatrixXd(4,4);
  const float noise_ax = 9, noise_ay = 9;
  float dt_2 = dt*dt;
  float dt_3 = dt_2*dt;
  float dt_4 = dt_3*dt;
  ekf_.Q_ << noise_ax*dt_4/4, 0, noise_ax*dt_3/2, 0,
             0, noise_ay*dt_4/4, 0, noise_ay*dt_3/2,
             noise_ax*dt_3/2, 0, dt_2*noise_ax, 0,
             0, noise_ay*dt_3/2, 0, noise_ay*dt_2;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  ekf_.H_ = H_laser_;
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    float r = measurement_pack.raw_measurements_[0];
    float phi = measurement_pack.raw_measurements_[1];
    float v = measurement_pack.raw_measurements_[2];
    VectorXd x_predicted(4);
    x_predicted << r*cos(phi), r*sin(phi), v*cos(phi), v*sin(phi);
    Hj_ =  tools.CalculateJacobian(x_predicted);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
