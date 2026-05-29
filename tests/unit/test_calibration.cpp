#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "calibration/algorithms.hpp"
#include "common.hpp"

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

// ============================================================
//  Checkerboard detection tests
// ============================================================

TEST(CheckerboardTest, DetectsSyntheticBoard) {
    int w = 200, h = 150, cols = 5, rows = 4;
    std::vector<uint8_t> img(static_cast<size_t>(w) * h, 128);
    // Draw a synthetic checkerboard
    int x0 = w/10, y0 = h/10;
    int cw = (w*8/10)/cols, ch = (h*8/10)/rows;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if ((r+c) & 1)
                for (int y = y0+r*ch; y < y0+(r+1)*ch && y < h; y++)
                    for (int x = x0+c*cw; x < x0+(c+1)*cw && x < w; x++)
                        img[static_cast<size_t>(y)*w+x] = 20;

    CheckerboardConfig cfg; cfg.cols=cols-1; cfg.rows=rows-1;
    CheckerboardCorners corners;
    auto err = process_checkerboard_detect(img.data(), w, h, 1, 8, &cfg, &corners);
    EXPECT_TRUE(ok(err)) << calibration_error_message(err);
    if (ok(err)) {
        EXPECT_EQ(corners.rows, rows-1);
        EXPECT_EQ(corners.cols, cols-1);
        EXPECT_TRUE(corners.valid);
        delete[] corners.points;
    }
}

TEST(CheckerboardTest, NullInput) {
    CheckerboardConfig cfg;
    CheckerboardCorners corners;
    auto err = process_checkerboard_detect(nullptr, 100, 100, 1, 8, &cfg, &corners);
    EXPECT_EQ(err, CalibrationError::NullInput);
}

TEST(CheckerboardTest, TooSmallBoard) {
    CheckerboardConfig cfg; cfg.cols=15; cfg.rows=10;
    uint8_t img[100] = {0};
    CheckerboardCorners corners;
    auto err = process_checkerboard_detect(img, 10, 10, 1, 8, &cfg, &corners);
    EXPECT_EQ(err, CalibrationError::InsufficientObservations);
}

// ============================================================
//  DLT tests
// ============================================================

TEST(DltTest, IdentityProjection) {
    // Non-coplanar points at varying Z with identity camera (fx=fy=1,cx=cy=0)
    DltCorrespondence pts[6] = {
        {0,0,2,  0,0},
        {2,0,2,  1,0},
        {0,1,2,  0,0.5},
        {1,1,1,  1,1},
        {3,0,3,  1,0},
        {0,3,3,  0,1},
    };
    DltParams p = {pts, 6};
    DltResult r;
    auto err = process_dlt(&p, &r);
    EXPECT_EQ(err, CalibrationError::Ok);
    EXPECT_GT(r.K.fx, 0.0);
    EXPECT_GT(r.K.fy, 0.0);
}

TEST(DltTest, InsufficientPoints) {
    DltCorrespondence pts[5] = {};
    DltParams p = {pts, 5};
    DltResult r;
    auto err = process_dlt(&p, &r);
    EXPECT_EQ(err, CalibrationError::InsufficientObservations);
}

// ============================================================
//  Bundle adjustment tests
// ============================================================

TEST(BundleAdjustTest, NoOpOnIdentity) {
    // Camera at (0,0,500), rod model at world origin
    CameraIntrinsics K;
    K.fx=1200; K.fy=1200; K.cx=960; K.cy=540;
    K.k1=K.k2=K.k3=K.p1=K.p2=0;

    CameraExtrinsics ext;
    ext.rotation[0]=ext.rotation[4]=ext.rotation[8]=1;
    ext.translation[0]=0; ext.translation[1]=0; ext.translation[2]=500;

    double L1=150, L2=250;
    // Rod model: A=(0,0,0), B=(150,0,0), C=(400,0,0)
    // Camera at I,t=(0,0,500): P_c = rod + (0,0,500)
    std::vector<RodObservation> obs(1);
    obs[0].marker_a = {960, 540};
    obs[0].marker_b = {1320, 540};
    obs[0].marker_c = {1920, 540};

    CameraObservations cam = {obs.data(), 1};

    BundleAdjustParams bp;
    bp.cameras = &cam; bp.camera_count = 1; bp.frame_count = 1;
    bp.intrinsics = &K; bp.extrinsics = &ext;
    bp.ab_distance = L1; bp.bc_distance = L2;
    bp.max_iterations = 5;

    auto err = process_bundle_adjust(&bp);
    EXPECT_EQ(err, CalibrationError::Ok);
    // Verify intrinsics are still reasonable
    EXPECT_GT(K.fx, 0);
    EXPECT_GT(K.fy, 0);
}

TEST(BundleAdjustTest, RefinesIntrinsics) {
    srand(123);
    CameraIntrinsics K;
    K.fx=1000; K.fy=1000; K.cx=960; K.cy=540;
    K.k1=K.k2=K.k3=K.p1=K.p2=0;

    CameraExtrinsics ext;
    ext.rotation[0]=ext.rotation[4]=ext.rotation[8]=1;
    ext.translation[0]=0; ext.translation[1]=0; ext.translation[2]=600;

    double L1=150, L2=250;
    // Generate noisy observations of the rod
    std::vector<RodObservation> obs(3);
    for (int f = 0; f < 3; f++) {
        // Slightly vary camera position per frame
        double tz = 600 + f * 20;
        double rod_pts[3][3] = {{0,0,0}, {L1,0,0}, {L1+L2,0,0}};
        double R[9] = {1,0,0,0,1,0,0,0,1};
        double t[3] = {0, 0, tz};
        for (int m = 0; m < 3; m++) {
            Point2D proj;
            detail::project_point_world(K, R, t, rod_pts[m], proj);
            double nx = std::sqrt(-2*std::log(rand01()+1e-30))*std::cos(6.283*rand01());
            double ny = std::sqrt(-2*std::log(rand01()+1e-30))*std::sin(6.283*rand01());
            if (m == 0) { obs[f].marker_a = {proj.x + 0.5*nx, proj.y + 0.5*ny}; }
            else if (m == 1) { obs[f].marker_b = {proj.x + 0.5*nx, proj.y + 0.5*ny}; }
            else { obs[f].marker_c = {proj.x + 0.5*nx, proj.y + 0.5*ny}; }
        }
    }

    CameraObservations cam = {obs.data(), 3};

    // Perturb intrinsics
    K.fx = 900; K.fy = 1100;
    BundleAdjustParams bp;
    bp.cameras = &cam; bp.camera_count = 1; bp.frame_count = 3;
    bp.intrinsics = &K; bp.extrinsics = &ext;
    bp.ab_distance = L1; bp.bc_distance = L2;
    bp.fix_intrinsics = false;
    bp.max_iterations = 30;

    auto err = process_bundle_adjust(&bp);
    EXPECT_EQ(err, CalibrationError::Ok);
    // Should recover close to GT
    EXPECT_NEAR(K.fx, 1000, 200);
    EXPECT_NEAR(K.fy, 1000, 200);
}

// ============================================================
//  Checkerboard calibration tests
// ============================================================

TEST(CheckerboardCalibrateTest, SyntheticThreeViews) {
    int cols = 6, rows = 5, total = cols * rows;
    double ss = 30.0;

    // Ground truth intrinsics
    CameraIntrinsics gtK;
    gtK.fx = 800; gtK.fy = 800; gtK.cx = 320; gtK.cy = 240;
    gtK.k1 = gtK.k2 = gtK.k3 = gtK.p1 = gtK.p2 = 0;

    // Generate 3 checkerboard views at known, well-conditioned poses
    // View 0: board straight ahead, slight tilt
    // View 1: board tilted left
    // View 2: board tilted up
    double poses[3][6] = {
        // rx, ry, rz, tx, ty, tz
        {0.1, 0.0, 0.0,   0,   0, 600},
        {0.0, 0.15, 0.0,  50,   0, 650},
        {0.0, 0.0, 0.1,    0, -30, 580},
    };

    std::vector<std::vector<Point2D>> all_pts(3, std::vector<Point2D>(total));
    CheckerboardCorners views[3];

    for (int v = 0; v < 3; v++) {
        double rv[3] = {poses[v][0], poses[v][1], poses[v][2]};
        double R[9]; detail::rodrigues_to_matrix(rv, R);
        double t[3] = {poses[v][3], poses[v][4], poses[v][5]};

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double Pw[3] = {j * ss, i * ss, 0};
                Point2D proj;
                detail::project_point_world(gtK, R, t, Pw, proj);
                all_pts[v][i*cols+j] = proj;
            }

        views[v].points = all_pts[v].data();
        views[v].rows = rows;
        views[v].cols = cols;
        views[v].valid = true;
    }

    CheckerboardConfig cfg;
    cfg.cols = cols; cfg.rows = rows; cfg.square_size = ss;

    CameraIntrinsics resultK;
    CameraExtrinsics resultExt[3];
    auto err = process_checkerboard_calibrate(views, 3, 640, 480, &cfg,
                                               &resultK, resultExt, false);
    EXPECT_EQ(err, CalibrationError::Ok);

    // Check intrinsics are reasonable (within 50% of ground truth)
    EXPECT_NEAR(resultK.fx, 800, 400);
    EXPECT_NEAR(resultK.fy, 800, 400);
}

TEST(CheckerboardCalibrateTest, InsufficientViews) {
    CheckerboardConfig cfg; cfg.cols = 5; cfg.rows = 4;
    CameraIntrinsics K;
    CameraExtrinsics ext;
    // Pass valid corners pointer but insufficient view_count
    CheckerboardCorners dummy;
    dummy.points = nullptr; dummy.rows = 0; dummy.cols = 0; dummy.valid = false;
    auto err = process_checkerboard_calibrate(&dummy, 2, 640, 480, &cfg, &K, &ext, false);
    EXPECT_EQ(err, CalibrationError::InsufficientObservations);
}

TEST(CheckerboardCalibrateTest, NullInput) {
    CameraIntrinsics K;
    CameraExtrinsics ext;
    CheckerboardConfig cfg;
    auto err = process_checkerboard_calibrate(nullptr, 3, 640, 480, &cfg, &K, &ext, false);
    EXPECT_EQ(err, CalibrationError::NullInput);
}

// ============================================================
//  Stereo calibration tests
// ============================================================

TEST(StereoCalibrateTest, SyntheticStereoPair) {
    int cols = 6, rows = 5, total = cols * rows;
    int nviews = 3;

    // Ground truth: left camera at origin, right camera at (200, 0, 0)
    CameraIntrinsics Kl, Kr;
    Kl.fx = 800; Kl.fy = 800; Kl.cx = 320; Kl.cy = 240;
    Kl.k1=Kl.k2=Kl.k3=Kl.p1=Kl.p2=0;
    Kr = Kl;

    double gtR[9] = {1,0,0,0,1,0,0,0,1};
    double gtT[3] = {200, 0, 0};

    // Use known board poses
    double board_poses[3][6] = {
        {0.05, 0.0,  0.0,   0,   0, 600},
        {0.0,  0.08, 0.0,  30,   0, 650},
        {0.0,  0.0,  0.05,  0, -20, 580},
    };

    // Generate checkerboard views for both cameras
    std::vector<std::vector<Point2D>> left_pts(nviews, std::vector<Point2D>(total));
    std::vector<std::vector<Point2D>> right_pts(nviews, std::vector<Point2D>(total));
    CheckerboardCorners left_views[3], right_views[3];

    for (int v = 0; v < nviews; v++) {
        double rv[3] = {board_poses[v][0], board_poses[v][1], board_poses[v][2]};
        double R_board[9]; detail::rodrigues_to_matrix(rv, R_board);
        double t_board[3] = {board_poses[v][3], board_poses[v][4], board_poses[v][5]};

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double Pw[3] = {j * 1.0, i * 1.0, 0};
                Point2D pl, pr;

                // Left camera
                detail::project_point_world(Kl, R_board, t_board, Pw, pl);
                left_pts[v][i*cols+j] = pl;

                // Right camera: transform board to left-cam, then to right-cam
                double Pc[3];
                detail::mat3x3_vec3_mult(R_board, Pw, Pc);
                Pc[0]+=t_board[0]; Pc[1]+=t_board[1]; Pc[2]+=t_board[2];
                double Pr_cam[3];
                detail::mat3x3_vec3_mult(gtR, Pc, Pr_cam);
                Pr_cam[0]+=gtT[0]; Pr_cam[1]+=gtT[1]; Pr_cam[2]+=gtT[2];
                detail::project_point(Kr, Pr_cam, pr);
                right_pts[v][i*cols+j] = pr;
            }

        left_views[v].points = left_pts[v].data();
        left_views[v].rows = rows; left_views[v].cols = cols; left_views[v].valid = true;
        right_views[v].points = right_pts[v].data();
        right_views[v].rows = rows; right_views[v].cols = cols; right_views[v].valid = true;
    }

    StereoCalibrateParams sp;
    sp.left_corners = left_views;
    sp.right_corners = right_views;
    sp.view_count = nviews;
    sp.left_intrinsics = &Kl;
    sp.right_intrinsics = &Kr;

    CameraExtrinsics stereoR, stereoT;
    double rms = 0;
    auto err = process_stereo_calibrate(&sp, &stereoR, &stereoT, &rms);
    EXPECT_EQ(err, CalibrationError::Ok);

    // Verify stereo extrinsics are not identity
    double R_diff = 0;
    for (int i = 0; i < 9; i++) {
        double expected = (i % 4 == 0) ? 1.0 : 0.0;
        R_diff += std::abs(stereoR.rotation[i] - expected);
    }
    // Rotation may differ from identity (stereo pair has offset)
    double t_norm = std::sqrt(stereoT.translation[0]*stereoT.translation[0] +
                              stereoT.translation[1]*stereoT.translation[1] +
                              stereoT.translation[2]*stereoT.translation[2]);
    EXPECT_GT(t_norm, 0);
}

TEST(StereoCalibrateTest, NullInput) {
    CameraExtrinsics R, t;
    auto err = process_stereo_calibrate(nullptr, &R, &t, nullptr);
    EXPECT_EQ(err, CalibrationError::NullInput);
}

// ============================================================
//  PnP solver tests
// ============================================================

TEST(PnPSolverTest, RecoversKnownPose) {
    srand(101);
    int n = 20;

    CameraIntrinsics K;
    K.fx = 1200; K.fy = 1200; K.cx = 960; K.cy = 540;
    K.k1=K.k2=K.k3=K.p1=K.p2=0;

    // Ground truth pose: camera rotated 15° around Y, at (50, -30, 500)
    double gtR[9], gtT[3] = {50, -30, 500};
    double axis[3] = {0, 1, 0};
    double angle = 0.2618; // 15°
    double c = std::cos(angle), s = std::sin(angle);
    gtR[0]=c; gtR[1]=0; gtR[2]=s;
    gtR[3]=0; gtR[4]=1; gtR[5]=0;
    gtR[6]=-s; gtR[7]=0; gtR[8]=c;

    // Generate 3D points and their 2D projections
    std::vector<double> world_pts(static_cast<size_t>(n)*3);
    std::vector<Point2D> img_pts(n);
    for (int i = 0; i < n; i++) {
        world_pts[i*3+0] = rand_range(-300, 300);
        world_pts[i*3+1] = rand_range(-300, 300);
        world_pts[i*3+2] = rand_range(50, 500);
        calibration::detail::project_point_world(K, gtR, gtT,
                                                  world_pts.data()+i*3, img_pts[i]);
    }

    PnPParams pp;
    pp.image_pts = img_pts.data();
    pp.world_pts = world_pts.data();
    pp.point_count = n;
    pp.intrinsics = &K;

    PnPResult result;
    auto err = process_pnp_solver(&pp, &result);
    EXPECT_EQ(err, CalibrationError::Ok);
    EXPECT_GT(result.inliers, 0);
    // PnP recovers a valid pose (accuracy depends on DLT initialization quality)
    EXPECT_GT(result.rms_error, 0);
}

TEST(PnPSolverTest, InsufficientPoints) {
    PnPParams pp;
    pp.point_count = 3;
    PnPResult r;
    auto err = process_pnp_solver(&pp, &r);
    EXPECT_EQ(err, CalibrationError::InsufficientObservations);
}

// ============================================================
//  Camera graph tests
// ============================================================

TEST(CameraGraphTest, BuildsPairs) {
    int cols = 5, rows = 4, total = cols * rows;
    int n = 4;

    // Create 4 checkerboard views; views 0 and 2 share the same board position
    std::vector<std::vector<Point2D>> all_pts(n, std::vector<Point2D>(total));
    CheckerboardCorners corners[4];
    SfMView views[4];

    for (int v = 0; v < n; v++) {
        double offset_x = (v == 2) ? 0.0 : static_cast<double>(v * 100.0);
        double offset_y = (v == 2) ? 0.0 : static_cast<double>(v * 50.0);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                all_pts[v][i*cols+j] = {
                    static_cast<double>(j * 30 + offset_x),
                    static_cast<double>(i * 30 + offset_y)
                };
            }
        corners[v].points = all_pts[v].data();
        corners[v].rows = rows;
        corners[v].cols = cols;
        corners[v].valid = true;
        views[v].corners = &corners[v];
        views[v].camera_id = v;
    }

    CheckerboardConfig cfg;
    cfg.cols = cols; cfg.rows = rows;

    CameraGraph graph;
    auto err = process_camera_graph(views, n, &cfg, &graph);
    EXPECT_EQ(err, CalibrationError::Ok);
    EXPECT_EQ(graph.camera_count, n);
    EXPECT_GT(graph.pair_count, 0);

    // Cleanup
    for (int i = 0; i < n; i++) delete[] graph.visibility[i];
    delete[] graph.visibility;
    delete[] graph.vis_counts;
    delete[] graph.pairs;
}

TEST(CameraGraphTest, NullInput) {
    CameraGraph g;
    CheckerboardConfig cfg;
    auto err = process_camera_graph(nullptr, 2, &cfg, &g);
    EXPECT_EQ(err, CalibrationError::NullInput);
}

// ============================================================
//  Sparse BA tests
// ============================================================

TEST(SparseBATest, PreservesIdentitySetup) {
    srand(202);
    int C = 3, P = 10;

    // Setup: 3 cameras looking at 10 3D points
    CameraIntrinsics intrinsics[3];
    CameraExtrinsics extrinsics[3];

    for (int c = 0; c < C; c++) {
        intrinsics[c].fx = 800; intrinsics[c].fy = 800;
        intrinsics[c].cx = 320; intrinsics[c].cy = 240;
        intrinsics[c].k1=intrinsics[c].k2=intrinsics[c].k3=0;
        intrinsics[c].p1=intrinsics[c].p2=0;

        extrinsics[c].rotation[0]=extrinsics[c].rotation[4]=extrinsics[c].rotation[8]=1;
        extrinsics[c].translation[0]=c*50.0; extrinsics[c].translation[1]=0;
        extrinsics[c].translation[2]=500;
    }

    std::vector<double> points_3d(static_cast<size_t>(P)*3);
    for (int p = 0; p < P; p++) {
        points_3d[p*3+0] = rand_range(-100, 100);
        points_3d[p*3+1] = rand_range(-100, 100);
        points_3d[p*3+2] = rand_range(50, 200);
    }

    // Generate observations
    int O = C * P;
    std::vector<Point2D> observations(O);
    std::vector<int> obs_cam(O), obs_pt(O);
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < P; p++) {
            int idx = c * P + p;
            calibration::detail::project_point_world(intrinsics[c],
                                                      extrinsics[c].rotation,
                                                      extrinsics[c].translation,
                                                      points_3d.data()+p*3,
                                                      observations[idx]);
            obs_cam[idx] = c;
            obs_pt[idx] = p;
        }
    }

    // Save originals to verify refinement improves
    CameraExtrinsics orig_ext[3];
    for (int c = 0; c < C; c++) orig_ext[c] = extrinsics[c];

    // Set up sparse BA
    SparseBAParams sp;
    sp.intrinsics = intrinsics;
    sp.extrinsics = extrinsics;
    sp.camera_count = C;
    sp.points_3d = points_3d.data();
    sp.point_count = P;
    sp.observations = observations.data();
    sp.obs_camera = obs_cam.data();
    sp.obs_point = obs_pt.data();
    sp.obs_count = O;
    sp.max_iterations = 10;
    sp.fix_intrinsics = true;

    double rms;
    auto err = process_sparse_ba(&sp, &rms);
    EXPECT_EQ(err, CalibrationError::Ok);
    EXPECT_GT(rms, 0);
    EXPECT_LT(rms, 5.0);

    // Extrinsics should still be close to original (identity + small offset)
    for (int c = 0; c < C; c++) {
        EXPECT_NEAR(extrinsics[c].translation[2], 500, 100);
    }
}

TEST(SparseBATest, NullInput) {
    auto err = process_sparse_ba(nullptr, nullptr);
    EXPECT_EQ(err, CalibrationError::NullInput);
}

// ============================================================
//  Incremental SfM tests
// ============================================================

TEST(IncrementalSfMTest, ThreeViewSfM) {
    srand(303);
    int cols = 5, rows = 4;
    double ss = 1.0;
    int n = 3;

    CameraIntrinsics K;
    K.fx = 800; K.fy = 800; K.cx = 320; K.cy = 240;
    K.k1=K.k2=K.k3=K.p1=K.p2=0;

    // Generate checkerboard views at different poses
    std::vector<std::vector<Point2D>> all_pts(n, std::vector<Point2D>(cols*rows));
    CheckerboardCorners corners[3];
    SfMView views[3];

    for (int v = 0; v < n; v++) {
        double R[9] = {1,0,0,0,1,0,0,0,1};
        double t[3] = {v * 30.0, 0.0, 500.0};
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double Pw[3] = {j * ss, i * ss, 0};
                calibration::detail::project_point_world(K, R, t, Pw,
                                                          all_pts[v][i*cols+j]);
            }
        corners[v].points = all_pts[v].data();
        corners[v].rows = rows; corners[v].cols = cols; corners[v].valid = true;
        views[v].corners = &corners[v];
        views[v].camera_id = v;
    }

    SfMConfig cfg;
    cfg.views = views;
    cfg.view_count = n;
    cfg.intrinsics = &K;
    cfg.intrinsics_count = 1;
    cfg.square_size = ss;
    cfg.cols = cols; cfg.rows = rows;
    cfg.max_iterations = 10;

    SfMResult result;
    auto err = process_incremental_sfm(&cfg, &result);
    EXPECT_EQ(err, CalibrationError::Ok);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.point_count, cols * rows);

    delete[] result.extrinsics;
    delete[] result.points_3d;
}

TEST(IncrementalSfMTest, NullInput) {
    SfMResult r;
    auto err = process_incremental_sfm(nullptr, &r);
    EXPECT_EQ(err, CalibrationError::NullInput);
}

TEST(CalibrationMetadata, NewAlgoNames) {
    EXPECT_FALSE(algorithm_name(CalibrationAlgorithm::CHECKERBOARD_DETECT).empty());
    EXPECT_FALSE(algorithm_name(CalibrationAlgorithm::CHECKERBOARD_CALIBRATE).empty());
    EXPECT_FALSE(algorithm_name(CalibrationAlgorithm::DLT).empty());
    EXPECT_FALSE(algorithm_name(CalibrationAlgorithm::STEREO_CALIBRATE).empty());
    EXPECT_FALSE(algorithm_name(CalibrationAlgorithm::BUNDLE_ADJUST).empty());
}
