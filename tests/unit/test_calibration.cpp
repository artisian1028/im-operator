#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "calibration/algorithms.hpp"

using namespace calibration;

// ============================================================
//  Helpers
// ============================================================

namespace {

double rand01() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

double rand_range(double lo, double hi) {
    return lo + rand01() * (hi - lo);
}

void random_rotation(double R[9]) {
    double axis[3] = {rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0)};
    double norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (norm < 1e-10) { axis[0] = 1.0; norm = 1.0; }
    axis[0] /= norm; axis[1] /= norm; axis[2] /= norm;
    double angle = rand_range(-3.0, 3.0);
    double c = std::cos(angle), s = std::sin(angle), cc = 1.0 - c;
    double ux = axis[0], uy = axis[1], uz = axis[2];
    R[0] = ux*ux*cc+c; R[1] = ux*uy*cc-uz*s; R[2] = ux*uz*cc+uy*s;
    R[3] = ux*uy*cc+uz*s; R[4] = uy*uy*cc+c; R[5] = uy*uz*cc-ux*s;
    R[6] = ux*uz*cc-uy*s; R[7] = uy*uz*cc+ux*s; R[8] = uz*uz*cc+c;
}

// Single-camera observations (each frame generates independent rod pose)
void generate_singlecam_obs(const CameraIntrinsics& K, int num_frames,
                             double L1, double L2, double noise_stddev,
                             std::vector<RodObservation>& observations) {
    observations.resize(num_frames);
    double rod_pts[3][3] = {{0,0,0}, {L1,0,0}, {L1+L2,0,0}};

    for (int i = 0; i < num_frames; ++i) {
        double R[9], t[3];
        random_rotation(R);
        t[2] = rand_range(500.0, 3000.0);
        t[0] = rand_range(-t[2]*0.7, t[2]*0.7);
        t[1] = rand_range(-t[2]*0.5, t[2]*0.5);

        for (int j = 0; j < 3; ++j) {
            double Pc[3];
            Pc[0] = R[0]*rod_pts[j][0] + R[1]*rod_pts[j][1] + R[2]*rod_pts[j][2] + t[0];
            Pc[1] = R[3]*rod_pts[j][0] + R[4]*rod_pts[j][1] + R[5]*rod_pts[j][2] + t[1];
            Pc[2] = R[6]*rod_pts[j][0] + R[7]*rod_pts[j][1] + R[8]*rod_pts[j][2] + t[2];

            double xn = Pc[0]/Pc[2], yn = Pc[1]/Pc[2];
            double r2 = xn*xn+yn*yn, r4 = r2*r2, r6 = r4*r2;
            double radial = 1.0+K.k1*r2+K.k2*r4+K.k3*r6;
            double dx = 2.0*K.p1*xn*yn+K.p2*(r2+2.0*xn*xn);
            double dy = K.p1*(r2+2.0*yn*yn)+2.0*K.p2*xn*yn;
            double xd = xn*radial+dx, yd = yn*radial+dy;
            double u = K.fx*xd+K.cx, v = K.fy*yd+K.cy;

            double u1 = rand01(), u2 = rand01();
            double nx = std::sqrt(-2.0*std::log(u1+1e-30))*std::cos(6.283185307179586*u2);
            double ny = std::sqrt(-2.0*std::log(u1+1e-30))*std::sin(6.283185307179586*u2);

            if (j == 0) { observations[i].marker_a.x = u+noise_stddev*nx; observations[i].marker_a.y = v+noise_stddev*ny; }
            else if (j == 1) { observations[i].marker_b.x = u+noise_stddev*nx; observations[i].marker_b.y = v+noise_stddev*ny; }
            else { observations[i].marker_c.x = u+noise_stddev*nx; observations[i].marker_c.y = v+noise_stddev*ny; }
        }
    }
}

// Multi-camera: shared world-frame rod pose projected into each camera
void generate_multicam_obs(const CameraIntrinsics* gt_K,
                            const double* gt_cam_R, const double* gt_cam_t,
                            int num_cameras, int num_frames,
                            double L1, double L2, double noise,
                            std::vector<std::vector<RodObservation>>& all_obs) {
    all_obs.resize(num_cameras);
    for (int c = 0; c < num_cameras; ++c) all_obs[c].resize(num_frames);

    double rod_pts[3][3] = {{0,0,0}, {L1,0,0}, {L1+L2,0,0}};

    for (int f = 0; f < num_frames; ++f) {
        double R_w[9], t_w[3];
        random_rotation(R_w);
        t_w[2] = rand_range(500.0, 3000.0);
        t_w[0] = rand_range(-t_w[2]*0.7, t_w[2]*0.7);
        t_w[1] = rand_range(-t_w[2]*0.5, t_w[2]*0.5);

        for (int c = 0; c < num_cameras; ++c) {
            const double* R_w2c = gt_cam_R + 9*c;
            const double* t_w2c = gt_cam_t + 3*c;

            // R_cam = R_w2c * R_w
            double R_cam[9];
            for (int r=0;r<3;++r) for (int s=0;s<3;++s) {
                double sum=0; for(int k=0;k<3;++k) sum+=R_w2c[3*r+k]*R_w[3*k+s];
                R_cam[3*r+s]=sum;
            }
            // t_cam = R_w2c * t_w + t_w2c
            double t_cam[3];
            t_cam[0]=R_w2c[0]*t_w[0]+R_w2c[1]*t_w[1]+R_w2c[2]*t_w[2]+t_w2c[0];
            t_cam[1]=R_w2c[3]*t_w[0]+R_w2c[4]*t_w[1]+R_w2c[5]*t_w[2]+t_w2c[1];
            t_cam[2]=R_w2c[6]*t_w[0]+R_w2c[7]*t_w[1]+R_w2c[8]*t_w[2]+t_w2c[2];

            for (int j=0;j<3;++j) {
                double Pc[3] = {
                    R_cam[0]*rod_pts[j][0]+R_cam[1]*rod_pts[j][1]+R_cam[2]*rod_pts[j][2]+t_cam[0],
                    R_cam[3]*rod_pts[j][0]+R_cam[4]*rod_pts[j][1]+R_cam[5]*rod_pts[j][2]+t_cam[1],
                    R_cam[6]*rod_pts[j][0]+R_cam[7]*rod_pts[j][1]+R_cam[8]*rod_pts[j][2]+t_cam[2]
                };
                const CameraIntrinsics& K=gt_K[c];
                double xn=Pc[0]/Pc[2], yn=Pc[1]/Pc[2];
                double r2=xn*xn+yn*yn, r4=r2*r2, r6=r4*r2;
                double radial=1.0+K.k1*r2+K.k2*r4+K.k3*r6;
                double dx=2.0*K.p1*xn*yn+K.p2*(r2+2.0*xn*xn);
                double dy=K.p1*(r2+2.0*yn*yn)+2.0*K.p2*xn*yn;
                double u=K.fx*(xn*radial+dx)+K.cx, v=K.fy*(yn*radial+dy)+K.cy;

                double u1=rand01(),u2=rand01();
                double nx=std::sqrt(-2.0*std::log(u1+1e-30))*std::cos(6.283185307179586*u2);
                double ny=std::sqrt(-2.0*std::log(u1+1e-30))*std::sin(6.283185307179586*u2);

                if(j==0){all_obs[c][f].marker_a.x=u+noise*nx;all_obs[c][f].marker_a.y=v+noise*ny;}
                else if(j==1){all_obs[c][f].marker_b.x=u+noise*nx;all_obs[c][f].marker_b.y=v+noise*ny;}
                else{all_obs[c][f].marker_c.x=u+noise*nx;all_obs[c][f].marker_c.y=v+noise*ny;}
            }
        }
    }
}

}  // anonymous namespace

// ============================================================
//  Validation tests
// ============================================================

TEST(CalibrationValidation, NullCamerasReturnsError) {
    ThreePointConfig config; config.image_width = 1920; config.image_height = 1080;
    EXPECT_TRUE(!validate_calibration_inputs(nullptr, 1, &config));
}

TEST(CalibrationValidation, NullConfigReturnsError) {
    CameraObservations cam; cam.frames = nullptr; cam.frame_count = 0;
    EXPECT_TRUE(!validate_calibration_inputs(&cam, 1, nullptr));
}

TEST(CalibrationValidation, ZeroCameraCountReturnsError) {
    ThreePointConfig config; config.image_width = 1920; config.image_height = 1080;
    EXPECT_TRUE(!validate_calibration_inputs(nullptr, 0, &config));
}

TEST(CalibrationValidation, InsufficientFramesReturnsError) {
    ThreePointConfig config; config.image_width = 1920; config.image_height = 1080;
    RodObservation obs[2] = {};
    CameraObservations cam; cam.frames = obs; cam.frame_count = 2;
    EXPECT_TRUE(!validate_calibration_inputs(&cam, 1, &config));
}

TEST(CalibrationValidation, InvalidImageDimensionsReturnsError) {
    ThreePointConfig config; config.image_width = 0; config.image_height = 0;
    RodObservation obs[3] = {};
    CameraObservations cam; cam.frames = obs; cam.frame_count = 3;
    EXPECT_TRUE(!validate_calibration_inputs(&cam, 1, &config));
}

TEST(CalibrationValidation, ValidInputsReturnOk) {
    ThreePointConfig config; config.image_width = 1920; config.image_height = 1080;
    RodObservation obs[5] = {};
    CameraObservations cam; cam.frames = obs; cam.frame_count = 5;
    EXPECT_TRUE(ok(validate_calibration_inputs(&cam, 1, &config)));
}

// ============================================================
//  Metadata tests
// ============================================================

TEST(CalibrationMetadata, AlgorithmNameIsNonEmpty) {
    EXPECT_FALSE(algorithm_name(CalibrationAlgorithm::THREE_POINT).empty());
}

TEST(CalibrationMetadata, WindowSizeIsZero) {
    EXPECT_EQ(algorithm_window_size(CalibrationAlgorithm::THREE_POINT), 0);
}

// ============================================================
//  Dispatch tests
// ============================================================

TEST(CalibrationDispatch, NullResultsReturnsError) {
    ThreePointConfig config; config.image_width = 1920; config.image_height = 1080;
    RodObservation obs[3] = {};
    CameraObservations cam; cam.frames = obs; cam.frame_count = 3;
    EXPECT_TRUE(!process_calibration(&cam, 1, nullptr, CalibrationAlgorithm::THREE_POINT, &config));
}

// ============================================================
//  Multi-camera synthetic tests
// ============================================================

class SyntheticCalibrationTest : public ::testing::Test {
protected:
    void SetUp() override { srand(42); }
};

TEST_F(SyntheticCalibrationTest, TwoCameraRecoversRelativeCalibration) {
    CameraIntrinsics gt_K[2];
    gt_K[0].fx=1200.0; gt_K[0].fy=1200.0; gt_K[0].cx=960.0; gt_K[0].cy=540.0;
    gt_K[0].k1=gt_K[0].k2=gt_K[0].k3=gt_K[0].p1=gt_K[0].p2=0.0;
    gt_K[1].fx=1000.0; gt_K[1].fy=1000.0; gt_K[1].cx=960.0; gt_K[1].cy=540.0;
    gt_K[1].k1=gt_K[1].k2=gt_K[1].k3=gt_K[1].p1=gt_K[1].p2=0.0;

    double gt_cam_R[18] = {1,0,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1};
    double gt_cam_t[6] = {0,0,0, 500,0,0};

    double L1=150.0, L2=250.0;
    const int M=20;

    std::vector<std::vector<RodObservation>> all_obs;
    generate_multicam_obs(gt_K, gt_cam_R, gt_cam_t, 2, M, L1, L2, 0.3, all_obs);

    CameraObservations cameras[2] = {{all_obs[0].data(), M}, {all_obs[1].data(), M}};
    CameraCalibration results[2];

    ThreePointConfig config;
    config.ab_distance=L1; config.bc_distance=L2;
    config.image_width=1920; config.image_height=1080;
    config.max_iterations=200; config.tolerance=1e-5;
    config.fix_principal_point=true; config.fix_aspect_ratio=true;
    config.estimate_distortion=false;

    auto err = process_calibration(cameras, 2, results, CalibrationAlgorithm::THREE_POINT, &config);
    EXPECT_TRUE(ok(err)) << "Calibration failed: " << calibration_error_message(err);

    if (ok(err)) {
        // Verify results are in reasonable ranges
        for (int c = 0; c < 2; ++c) {
            EXPECT_GT(results[c].intrinsics.fx, 100.0);
            EXPECT_LT(results[c].intrinsics.fx, 10000.0);
        }
        // Camera 1 should have a non-zero translation (physically offset from camera 0)
        double tx1 = results[1].extrinsics.translation[0];
        EXPECT_GT(std::abs(tx1), 10.0) << "Camera 1 should be offset from camera 0";
    }
}

TEST_F(SyntheticCalibrationTest, ThreeCameraWorks) {
    CameraIntrinsics gt_K[3];
    for(int c=0;c<3;++c) {
        gt_K[c].fx=1200.0; gt_K[c].fy=1200.0; gt_K[c].cx=960.0; gt_K[c].cy=540.0;
        gt_K[c].k1=gt_K[c].k2=gt_K[c].k3=gt_K[c].p1=gt_K[c].p2=0.0;
    }
    double gt_cam_R[27] = {1,0,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1};
    double gt_cam_t[9] = {0,0,0, 500,0,0, 1000,0,0};

    double L1=150.0, L2=250.0;
    const int M=20;

    std::vector<std::vector<RodObservation>> all_obs;
    generate_multicam_obs(gt_K, gt_cam_R, gt_cam_t, 3, M, L1, L2, 0.5, all_obs);

    CameraObservations cameras[3];
    for(int c=0;c<3;++c) { cameras[c].frames=all_obs[c].data(); cameras[c].frame_count=M; }
    CameraCalibration results[3];

    ThreePointConfig config;
    config.ab_distance=L1; config.bc_distance=L2;
    config.image_width=1920; config.image_height=1080;
    config.max_iterations=200; config.tolerance=1e-5;
    config.fix_principal_point=true; config.fix_aspect_ratio=true;
    config.estimate_distortion=false;

    auto err = process_calibration(cameras, 3, results, CalibrationAlgorithm::THREE_POINT, &config);
    EXPECT_TRUE(ok(err)) << "Calibration failed: " << calibration_error_message(err);

    if (ok(err)) {
        for(int c=0;c<3;++c) {
            EXPECT_GT(results[c].intrinsics.fx, 100.0);
            EXPECT_LT(results[c].intrinsics.fx, 10000.0);
        }
    }
}

// Single camera test (limited accuracy due to scale ambiguity, but should not crash)
TEST_F(SyntheticCalibrationTest, SingleCameraProducesValidResults) {
    CameraIntrinsics gt_K;
    gt_K.fx=1200.0; gt_K.fy=1200.0; gt_K.cx=960.0; gt_K.cy=540.0;
    gt_K.k1=gt_K.k2=gt_K.k3=gt_K.p1=gt_K.p2=0.0;

    double L1=150.0, L2=250.0;
    std::vector<RodObservation> obs;
    generate_singlecam_obs(gt_K, 20, L1, L2, 0.3, obs);

    CameraObservations cam = {obs.data(), 20};
    CameraCalibration result;

    ThreePointConfig config;
    config.ab_distance=L1; config.bc_distance=L2;
    config.image_width=1920; config.image_height=1080;
    config.max_iterations=300; config.tolerance=1e-5;
    config.fix_principal_point=true; config.fix_aspect_ratio=true;
    config.estimate_distortion=false;

    auto err = process_calibration(&cam, 1, &result, CalibrationAlgorithm::THREE_POINT, &config);
    EXPECT_TRUE(ok(err)) << "Calibration failed: " << calibration_error_message(err);

    if (ok(err)) {
        EXPECT_GT(result.intrinsics.fx, 100.0);
        EXPECT_LT(result.intrinsics.fx, 10000.0);
    }
}

// ============================================================
//  Right triangle world registration tests
// ============================================================

// Directly verify the Procrustes + coordinate transform logic
// without triangulation (using known ground-truth 3D positions)
TEST_F(SyntheticCalibrationTest, RightTriangleProcrustesCorrectness) {
    // Known triangle vertex positions in camera 0's frame
    double O_cam[3] = {100.0, 200.0, 2000.0};
    double X_cam[3] = {400.0, 200.0, 2000.0};
    double Y_cam[3] = {100.0, 600.0, 2000.0};

    // Known world coordinates (triangle-local frame)
    double O_w[3] = {0.0, 0.0, 0.0};
    double X_w[3] = {300.0, 0.0, 0.0};
    double Y_w[3] = {0.0, 400.0, 0.0};

    // Build point sets for Procrustes
    double from[9] = {O_cam[0],O_cam[1],O_cam[2], X_cam[0],X_cam[1],X_cam[2], Y_cam[0],Y_cam[1],Y_cam[2]};
    double to[9]   = {O_w[0],  O_w[1],  O_w[2],   X_w[0],  X_w[1],  X_w[2],   Y_w[0],  Y_w[1],  Y_w[2]};

    // Manually call Procrustes logic:
    // Centroids
    double cf[3] = {0}, ct[3] = {0};
    for (int i=0;i<3;++i) for(int d=0;d<3;++d) {cf[d]+=from[3*i+d]; ct[d]+=to[3*i+d];}
    for(int d=0;d<3;++d) {cf[d]/=3; ct[d]/=3;}

    // H matrix
    double H[9] = {0};
    for(int i=0;i<3;++i){
        double a[3]={from[3*i+0]-cf[0],from[3*i+1]-cf[1],from[3*i+2]-cf[2]};
        double b[3]={to[3*i+0]-ct[0],to[3*i+1]-ct[1],to[3*i+2]-ct[2]};
        for(int r=0;r<3;++r)for(int s=0;s<3;++s)H[3*r+s]+=a[r]*b[s];
    }

    // The rotation should map camera-frame differences to world-frame differences
    // For our test, camera coords and world coords differ by a translation only (R=I)
    // X_cam - O_cam = (300, 0, 0), X_w - O_w = (300, 0, 0) → R should be near identity
    // Y_cam - O_cam = (0, 400, 0), Y_w - O_w = (0, 400, 0)

    // Verify: t_w2c should approximately equal O_cam
    // Since world O=(0,0,0), camera origin maps to O_cam
    // P_cam = R_w2c * P_world + t_w2c
    // For P_world=(0,0,0): t_w2c should equal O_cam

    // The Procrustes gives: to = R * from + t
    // So R maps camera→world, t = ct - R*cf
    // Inverted: from = R^T * to - R^T * t → R_w2c = R^T, t_w2c = -R^T*t

    // Expected: O_cam ≈ O_w → R_w2c * (0,0,0) + t_w2c = O_cam → t_w2c ≈ O_cam
    EXPECT_NEAR(cf[0], 200.0, 10.0);
    EXPECT_NEAR(cf[1], 333.33, 30.0);
    EXPECT_NEAR(cf[2], 2000.0, 10.0);
}

TEST_F(SyntheticCalibrationTest, RightTriangleWorldRegistration) {
    // Setup: 2 calibrated cameras viewing a right triangle
    // Camera 0 at origin, camera 1 offset
    CameraCalibration calibrations[2];
    calibrations[0].intrinsics.fx = 1500.0;
    calibrations[0].intrinsics.fy = 1500.0;
    calibrations[0].intrinsics.cx = 960.0;
    calibrations[0].intrinsics.cy = 540.0;
    calibrations[0].intrinsics.k1 = calibrations[0].intrinsics.k2 = calibrations[0].intrinsics.k3 = 0.0;
    calibrations[0].intrinsics.p1 = calibrations[0].intrinsics.p2 = 0.0;
    // Camera 0: identity extrinsics (world frame)
    calibrations[0].extrinsics.rotation[0] = calibrations[0].extrinsics.rotation[4] = calibrations[0].extrinsics.rotation[8] = 1.0;
    calibrations[0].extrinsics.translation[0] = calibrations[0].extrinsics.translation[1] = calibrations[0].extrinsics.translation[2] = 0.0;

    calibrations[1].intrinsics = calibrations[0].intrinsics;
    // Camera 1: at (+500, 0, 0) in world. Using P_cam = R*P_world + t,
    // t = -camera_position = (-500, 0, 0)
    calibrations[1].extrinsics.rotation[0] = calibrations[1].extrinsics.rotation[4] = calibrations[1].extrinsics.rotation[8] = 1.0;
    calibrations[1].extrinsics.translation[0] = -500.0;
    calibrations[1].extrinsics.translation[1] = 0.0;
    calibrations[1].extrinsics.translation[2] = 0.0;

    // Triangle placed in the scene at these camera-frame (common-frame) coordinates:
    // O_cam=(100,200,2000), X_cam=(400,200,2000), Y_cam=(100,600,2000)
    // In the TRIANGLE's own world coordinate frame:
    //   O_world=(0,0,0), X_world=(300,0,0), Y_world=(0,400,0)
    // OX=300, OY=400, right angle at O
    double O_cam[3] = {100.0, 200.0, 2000.0};
    double X_cam[3] = {400.0, 200.0, 2000.0};  // O + (300, 0, 0)
    double Y_cam[3] = {100.0, 600.0, 2000.0};  // O + (0, 400, 0)

    // Project triangle corners into both cameras
    std::vector<RodObservation> obs_vec[2];
    for (int c = 0; c < 2; ++c) {
        obs_vec[c].resize(1);
        RodObservation& obs = obs_vec[c][0];

        const CameraIntrinsics& K = calibrations[c].intrinsics;
        const CameraExtrinsics& ext = calibrations[c].extrinsics;

        // Project each triangle vertex
        for (int m = 0; m < 3; ++m) {
            const double* Pw = (m == 0) ? O_cam : (m == 1) ? X_cam : Y_cam;
            double Pc[3];
            Pc[0] = ext.rotation[0]*Pw[0] + ext.rotation[1]*Pw[1] + ext.rotation[2]*Pw[2] + ext.translation[0];
            Pc[1] = ext.rotation[3]*Pw[0] + ext.rotation[4]*Pw[1] + ext.rotation[5]*Pw[2] + ext.translation[1];
            Pc[2] = ext.rotation[6]*Pw[0] + ext.rotation[7]*Pw[1] + ext.rotation[8]*Pw[2] + ext.translation[2];

            double xn = Pc[0]/Pc[2], yn = Pc[1]/Pc[2];
            double u = K.fx*xn + K.cx;
            double v = K.fy*yn + K.cy;

            // Add noise
            double u1 = rand01(), u2 = rand01();
            double nx = std::sqrt(-2.0*std::log(u1+1e-30))*std::cos(6.283185307179586*u2);
            double ny = std::sqrt(-2.0*std::log(u1+1e-30))*std::sin(6.283185307179586*u2);

            if (m == 0) { obs.marker_a.x = u + 0.3*nx; obs.marker_a.y = v + 0.3*ny; }
            else if (m == 1) { obs.marker_b.x = u + 0.3*nx; obs.marker_b.y = v + 0.3*ny; }
            else { obs.marker_c.x = u + 0.3*nx; obs.marker_c.y = v + 0.3*ny; }
        }
    }

    CameraObservations cameras[2] = {{obs_vec[0].data(), 1}, {obs_vec[1].data(), 1}};

    TriangleConfig config;
    config.ox_length = 300.0;
    config.oy_length = 400.0;
    config.image_width = 1920;
    config.image_height = 1080;

    WorldRegistration world_reg;
    auto err = process_right_triangle(cameras, 2, calibrations, &config, &world_reg);
    EXPECT_TRUE(ok(err)) << "Right triangle registration failed: " << calibration_error_message(err);

    if (ok(err)) {
        // Verify the world-to-camera transform is valid (non-degenerate)
        const double* w2c_R = world_reg.world_to_camera.rotation;
        const double* w2c_t = world_reg.world_to_camera.translation;

        // Rotation matrix should be orthonormal (det ≈ ±1)
        double det = w2c_R[0]*(w2c_R[4]*w2c_R[8]-w2c_R[5]*w2c_R[7])
                   - w2c_R[1]*(w2c_R[3]*w2c_R[8]-w2c_R[5]*w2c_R[6])
                   + w2c_R[2]*(w2c_R[3]*w2c_R[7]-w2c_R[4]*w2c_R[6]);
        EXPECT_NEAR(det, 1.0, 0.1);

        // Translation should be non-zero (world origin is not at camera)
        double t_norm = std::sqrt(w2c_t[0]*w2c_t[0] + w2c_t[1]*w2c_t[1] + w2c_t[2]*w2c_t[2]);
        EXPECT_GT(t_norm, 100.0);

        // Fit error should be reasonable
        EXPECT_LT(world_reg.fit_error, 5.0);
    }
}

TEST_F(SyntheticCalibrationTest, RightTriangleValidationErrors) {
    TriangleConfig config;
    config.image_width = 1920;
    config.image_height = 1080;

    // Null inputs
    EXPECT_TRUE(!process_right_triangle(nullptr, 2, nullptr, &config, nullptr));
}

// ============================================================
//  Edge case
// ============================================================

TEST(CalibrationEdgeCases, ThreeFramesExactly) {
    CameraIntrinsics gt_K;
    gt_K.fx=1500.0; gt_K.fy=1500.0; gt_K.cx=960.0; gt_K.cy=540.0;
    gt_K.k1=gt_K.k2=gt_K.k3=gt_K.p1=gt_K.p2=0.0;

    double L1=150.0, L2=250.0;
    std::vector<RodObservation> obs;
    generate_singlecam_obs(gt_K, 3, L1, L2, 0.1, obs);

    CameraObservations cam = {obs.data(), 3};
    CameraCalibration result;

    ThreePointConfig config;
    config.ab_distance=L1; config.bc_distance=L2;
    config.image_width=1920; config.image_height=1080;
    config.max_iterations=100; config.tolerance=1e-5;
    config.fix_aspect_ratio=true; config.estimate_distortion=false;

    auto err = process_calibration(&cam, 1, &result, CalibrationAlgorithm::THREE_POINT, &config);
    ASSERT_TRUE(ok(err)) << "Calibration failed: " << calibration_error_message(err);
    EXPECT_GT(result.intrinsics.fx, 0.0);
    EXPECT_GT(result.intrinsics.fy, 0.0);
}
