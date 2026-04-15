// Copyright (c) HEX lab, Technical University of Munich

#include <filesystem/filesystem.h>
#include <m3t/azure_kinect_multi.h>
#include <m3t/hololens_camera.h>
#include <m3t/basic_depth_renderer.h>
#include <m3t/body.h>
#include <m3t/common.h>
#include <m3t/depth_modality.h>
#include <m3t/depth_model.h>
#include <m3t/link.h>
#include <m3t/normal_viewer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/renderer_geometry.h>
#include <m3t/static_detector.h>
#include <m3t/texture_modality.h>
#include <m3t/tracker.h>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <mmdeploy/pose_detector.h>
#include <pangolin/pangolin.h>
#include <torch/torch.h>
#include "cosypose.hpp"
#include <ranges>
#include <conio.h>
#include <windows.h>

std::unordered_map<std::string, std::vector<std::vector<cv::Point3f>>> kpts_gt_map;
//std::unordered_map<std::string, std::vector<std::string>> labels_map;
std::unordered_map<std::string, std::vector<std::vector<m3t::Transform3fA>>> symmetries_map;
std::unordered_map<std::string, std::unordered_map<int, int>> n_sym_mapping_all;
torch::Tensor sym_tensor, kpts_3d;
std::unordered_map<std::string, std::vector<int>> symmetry_ids_map;
int n_sym = 64;
std::unordered_map<std::string, cosypose::Mesh_database> mesh_db_map;
std::unordered_map<std::string, std::map<std::string, int>> body_name2idx_map_all{};
std::shared_ptr<hl2ss::ipc_umq> client;
hl2ss::umq_command_buffer buffer;

std::string object_name = "nailing";
std::string data_root_dir = "../../data/";
std::string model_root_dir = "path/to/checkpoint_multicam";
int num_cam = 1;
//std::string model_file_path = data_root_dir + object_name + "/kpts";

//const std::filesystem::path directory{root_dir + object_name + "/models"};
//std::filesystem::path model_path{model_root_dir + "checkpoint_" + object_name};
auto device_name = "cuda";
int flag = 0;
int flag_send = 0;

std::map<std::string, std::vector<std::string>> body_names_map = {
    {"nailing",
      {
         "Nailing_aimingarm",
         "Nailing_bar",
         "Nailing_gear",
         "Nailing_insertionhandle",
         "Nailing_nail",
         "Nailing_roller",
         "Nailing_screw",
         "Nailing_spacer",
         "Nailing_stick",
      }
    }
};

std::vector<std::string> obj_list = { "nailing"};

std::string json = "[";

Eigen::Matrix<float, 6, 6> Adjoint(const m3t::Transform3fA &pose) {
  Eigen::Matrix<float, 6, 6> m{Eigen::Matrix<float, 6, 6>::Zero()};
  m.topLeftCorner<3, 3>() = pose.rotation();
  m.bottomLeftCorner<3, 3>() =
      m3t::Vector2Skewsymmetric(pose.translation()) * pose.rotation();
  m.bottomRightCorner<3, 3>() = pose.rotation();
  return m;
}

void Tensor2EigenMat(torch::Tensor &t, Eigen::MatrixXf &m) {
  if (t.dim() == 2) {
    Eigen::MatrixXf m_tmp(t.size(1), t.size(0));
    auto num_elements = t.numel();
    auto element_size = t.element_size();
    auto size_in_bytes = num_elements * element_size;
    std::copy(t.data_ptr<float>(), t.data_ptr<float>() + t.numel(),
              m_tmp.data());
    m = m_tmp.transpose();
  } else {
    std::cerr << "t.dim()=" << t.dim();
  }
}

void EigenMat2Tensor(torch::Tensor &t, Eigen::MatrixXf &x) {
  t = torch::from_blob(x.data(), {x.cols(), x.rows()}, torch::kFloat32).t();
}

torch::Tensor pad_stack_tensors(
    std::vector<std::vector<m3t::Transform3fA>> vec_list) {
  int bsz = vec_list.size();
  int n_max = 0;

  for (auto vec_n : vec_list) {
    if (vec_n.size() > n_max) n_max = vec_n.size();
  }
  // creat an empty tensor in the shape of (bsz, n_max, 4, 4)
  torch::Tensor tensor_padded = torch::zeros({bsz, n_max, 4, 4});

  // tensor_list_padded = [];
  for (int vec_id = 0; vec_id < vec_list.size(); vec_id++) {
    auto vec_n = vec_list[vec_id];
    int n_pad = n_max - vec_n.size();
    if (n_pad > 0) {
      std::vector<m3t::Transform3fA> pad;

      std::vector<int> prob(vec_n.size(), 1);
      // generate non-uniform distribution (default result_type is int)
      std::discrete_distribution const distribution{prob.begin(), prob.end()};
      std::vector<decltype(distribution)::result_type> indices;
      indices.reserve(n_pad);  // reserve to prevent reallocation
      std::generate_n(
          std::back_inserter(indices), n_pad,
          [distribution =
               std::move(distribution),  // could also capture by reference (&)
                                         // or construct in the capture list
           generator = std::default_random_engine{0}
           // pseudo random. Fixed seed! Always same output.
      ]() mutable {  // mutable required for generator
            return distribution(generator);
          });
      pad.reserve(n_pad);
      std::transform(cbegin(indices), cend(indices), std::back_inserter(pad),
                     [&vec_n](auto const index) {
                       return *std::next(cbegin(vec_n), index);
                       // note, for std::vector or std::array of samples, you
                       // can use return samples[index];
                     });
      vec_n.insert(vec_n.end(), pad.begin(), pad.end());
    }
    for (int id = 0; id < vec_n.size(); id++) {
      auto vec = vec_n[id].matrix();
      torch::Tensor T;
      Eigen::MatrixXf mat = vec;
      EigenMat2Tensor(T, mat);
      tensor_padded.index({vec_id, id}) = T;
    }
  }
  return tensor_padded;
}

void DrawCamera(const Eigen::Matrix4d &Tcw,
                Eigen::Vector3d color = Eigen::Vector3d(0, 1, 0)) {
  const float size = 0.1;
  glPushMatrix();
  glMultMatrixd(Tcw.data());
  // Draw contour of camera
  glLineWidth(2);
  glBegin(GL_LINES);

  glColor3f(color[0], color[1], color[2]);
  glVertex3f(0, 0, 0);
  glVertex3f(size, size, size);

  glVertex3f(0, 0, 0);
  glVertex3f(size, -size, size);

  glVertex3f(0, 0, 0);
  glVertex3f(-size, -size, size);

  glVertex3f(0, 0, 0);
  glVertex3f(-size, size, size);

  glVertex3f(size, size, size);
  glVertex3f(size, -size, size);

  glVertex3f(size, -size, size);
  glVertex3f(-size, -size, size);

  glVertex3f(-size, -size, size);
  glVertex3f(-size, size, size);

  glVertex3f(-size, size, size);
  glVertex3f(size, size, size);
  glEnd();

  glPopMatrix();
}

void DrawPoint(const Eigen::Vector3d &landmark,
               const Eigen::Vector3d color = Eigen::Vector3d(1, 0, 0)) {
  glPointSize(5);
  glBegin(GL_POINTS);
  glColor3f(color[0], color[1], color[2]);
  glVertex3d(landmark[0], landmark[1], landmark[2]);
  glEnd();
}


void DrawTrajectory(
    std::shared_ptr<m3t::HololensColorCamera> hololens_color_ptr,
    std::vector<std::shared_ptr<m3t::AzureKinectColorMulti>> color_camera_ptrs,
    std::shared_ptr<std::vector<m3t::Transform3fA>> body_pose_ptr) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);  // create window

  glEnable(GL_DEPTH_TEST);                            // begin depth test
  glEnable(GL_BLEND);                                 // begin blending
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // set blend function

  auto rotation = hololens_color_ptr->camera2world_pose().rotation();
  auto translation = hololens_color_ptr->camera2world_pose().translation();
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1,
                                 1000),  //projection matrix
      // screen width, height, horizontal view, vertical view, z axis postion, min and max value from camera to screen
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0,
                                0.0)  // view matrix
      //camera position, observation point, observation vector
  );

  pangolin::View &d_cam =
      pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
          // start and end point of window in x and y axis, width height ratio, negative means 768：1024
          .SetHandler(new pangolin::Handler3D(s_cam));
  Eigen::Vector3f Ow, Xw, Yw, Zw;
  std::vector<Eigen::Vector3f> trans;
  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();
  while (true) {
    //std::cout << "set up" << std::endl;
    glClear(GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT);  // clear color and depth buffer
    Twc = hololens_color_ptr->camera2world_pose().matrix();
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);         // activate visualzation and rendering
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // set window color
    glLineWidth(2);                        // set line width
    //std::cout << "get pose" << std::endl;
      // vis coordinates of each pose
    // Visualize cameras
    for (int cam_id = 0; cam_id < color_camera_ptrs.size(); cam_id++) {
      if (!color_camera_ptrs[cam_id]->camera2world_pose().translation().isZero(0))
      DrawCamera(
          color_camera_ptrs[cam_id]->camera2world_pose().matrix().cast<double>());
    }

    // Visualize object center points
    //std::cout << body_pose_ptr->size() << std::endl;
    for (int obj_id = 0; obj_id < body_pose_ptr->size(); obj_id++) {
      if (!body_pose_ptr->at(obj_id).translation().isZero(
              0))
        DrawPoint(body_pose_ptr->at(obj_id).translation().cast<double>());
    }
    try {
      DrawCamera(hololens_color_ptr->camera2world_pose().matrix().cast<double>());
      Ow = hololens_color_ptr->camera2world_pose()
               .translation();  // get camera pose
      auto rotation = hololens_color_ptr->camera2world_pose().rotation();
      Xw =
          Ow +
          0.1 *
              Eigen::Vector3f(
                  rotation(0, 0), rotation(1, 0),
                  rotation(
                      2,
                      0));  // get unit vector of x axis, scale by 0.1
      Yw = Ow + 0.1 * Eigen::Vector3f(rotation(0, 1), rotation(1, 1),
                                      rotation(2,
                                               1));  // get unit vector of y axis
      Zw = Ow + 0.1 * Eigen::Vector3f(rotation(0, 2), rotation(1, 2),
                                      rotation(2,
                                               2));  // get unit vector of z axis
    } catch (...) {
    }
      
      //std::cout << "begin visualize" << std::endl;
      glBegin(GL_LINES);         // start draw lines
      glColor3f(1.0, 0.0, 0.0);  // set rgb
      //draw two end points of line
      glVertex3f(Ow[0], Ow[1], Ow[2]);  // origin coordinate
      glVertex3f(Xw[0], Xw[1],
                 Xw[2]);  // coordinate of x axis    ----> red

      glColor3f(0.0, 1.0, 0.0);
      glVertex3f(Ow[0], Ow[1], Ow[2]);  // origin coordinate
      glVertex3f(Yw[0], Yw[1],
                 Yw[2]);  // coordinate of y axis    ----> green

      glColor3f(0.0, 0.0, 1.0);
      glVertex3f(Ow[0], Ow[1], Ow[2]);  // origin coordinate
      glVertex3f(Zw[0], Zw[1],
                 Zw[2]);  // cooridnate of z axis    ----> blue
      glEnd();            // end painting
    

      // draw lines
      if (trans.size()<2000) {
         trans.push_back(Ow);
      } else {
        trans.erase(trans.begin());
        trans.push_back(Ow);

      }
      if (trans.size() > 2) {
            for (size_t i = 0; i < trans.size()-1; i++) {
              glColor3f(0.0, 0.0, 1.0);               // black
              glBegin(GL_LINES);                      // start visualize lines
              auto p1 = trans[i], p2 = trans[i + 1];  // get camera poses in consecutive frames
              glVertex3f(
                  p1[0], p1[1],
                  p1[2]);  // visulize two end points of the line(3d camera position in neighbor)
              glVertex3f(p2[0], p2[1], p2[2]);
              glEnd();
            }
      }

    pangolin::FinishFrame();  // end visualizing of current frame
    //std::cout << "end visualize" << std::endl;
    _sleep(5);             // sleep 5ms ，visualize every 5 ms
  }
}


void tcpserver() {
  try {
    client->push(buffer.get_data(), buffer.get_size());
    std::vector<uint32_t> response;
    response.resize(buffer.get_count());
    client->pull(response.data(), buffer.get_count());
    std::cout << response[0] << std::endl;
  }
  catch (...) {
    std::cerr << "server connection failed" << std::endl;
  }
}

    
void UpdatePublisher() {
  // send pose information back to HOlOLENS
  while (true) {
    if (flag == 1) flag_send = 1;
    std::string json_send = json;
    std::string name("FootPedal");
    json_send += "{  \"classId\": \"" + name +
            "\" , \"x\": " + std::to_string(flag_send) + "\ , \"y\": " + "0" +
            "\ , \"z\": " + "0" + "\ , \"qx\": " + "0" + "\ , \"qy\": " + "0" +
            "\ , \"qz\": " + "0" + "\},";
    json_send[json_send.size() - 1] = ']';
    std::vector<uint8_t> data(json_send.begin(), json_send.end());
    buffer.clear();
    buffer.add(0xFFFFFFFE, data.data(), data.size());
    client->push(buffer.get_data(), buffer.get_size());
    std::vector<uint32_t> response;
    response.resize(buffer.get_count());
    client->pull(response.data(), buffer.get_count());
    int obj_id = response[0];
    object_name = obj_list[obj_id];
    std::cout << obj_id << std::endl;
    // std::thread server(tcpserver);
    // server.detach();
    flag_send = 0;
  }

  //return true;
}

void ExecuteTracking(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    std::vector<std::shared_ptr<m3t::AzureKinectColorMulti>> &color_camera_ptrs,
    std::vector<std::shared_ptr<m3t::AzureKinectDepthMulti>> &depth_camera_ptrs,
    std::vector<std::shared_ptr<m3t::Link>> link_ptrs,
    std::vector<std::vector<std::vector<std::shared_ptr<m3t::Modality>>>> cam_link_ptr,
                     int iteration) {
  // Update Cameras
  tracker_ptr->UpdateCameras(iteration);
  
  for (int corr_iteration = 0;
       corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {
    // Calculate correspondences
    
    tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);
    // Visualize correspondences
    int corr_save_idx =
        iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
    tracker_ptr->VisualizeCorrespondences(corr_save_idx);
    
    //std::cout << tracker_ptr->n_corr_iterations() << std::endl;
    for (int update_iteration = 0;
         update_iteration < tracker_ptr->n_corr_iterations();
         ++update_iteration) {
      // Calculate gradient and hessian
      tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration,
                                               update_iteration);
      //std::cout << cam_link_ptr[0][0][0]->gradient().transpose() << std::endl;
      
      
      // Calculate optimization
      tracker_ptr->CalculateOptimization(iteration, corr_iteration,
                                         update_iteration);
      
      for (int cam_id = 0; cam_id < color_camera_ptrs.size(); cam_id++) {
        Eigen::VectorXf b{Eigen::VectorXf::Zero(6)};
        Eigen::MatrixXf a{Eigen::MatrixXf::Zero(6, 6)};
        for (int obj_id = 0; obj_id < link_ptrs.size(); obj_id++) {
          Eigen::Matrix<float, 6, 1> gradient{};
          Eigen::Matrix<float, 6, 6> hessian{};
          gradient.setZero();
          hessian.setZero();
          
          m3t::Transform3fA obj2cam_pose =
              color_camera_ptrs[cam_id]->camera2world_pose().inverse() *
              link_ptrs[obj_id]->link2world_pose();
          
          auto jacobian{Adjoint(obj2cam_pose)};
          for (auto &modality_ptr : cam_link_ptr[cam_id][obj_id]) {
            gradient -= 0.01*modality_ptr->gradient();
            hessian += modality_ptr->hessian();
          }
          // jacobian.transpose() * gradient;
          b += jacobian.transpose() * gradient;
          a.triangularView<Eigen::Lower>() -=
              jacobian.transpose() * hessian * jacobian;
        }
        Eigen::LDLT<Eigen::MatrixXf, Eigen::Lower> ldlt{a};
        Eigen::VectorXf theta{ldlt.solve(b)};
        //std::cout << theta.transpose() << std::endl;
        if (theta.array().isNaN().isZero()) {
          // Calculate pose variation
          m3t::Transform3fA pose_variation{m3t::Transform3fA::Identity()};
          pose_variation.translate(theta.tail<3>());
          pose_variation.rotate(
              m3t::Vector2Skewsymmetric(theta.head<3>()).exp());
          if (color_camera_ptrs[cam_id]
                  ->camera2world_pose()
                  .matrix()
                  .array()
                  .isNaN()
                  .isZero()) {
            color_camera_ptrs[cam_id]->set_camera2world_pose(
                pose_variation *
                color_camera_ptrs[cam_id]->camera2world_pose());
            m3t::Transform3fA depth_pose;
            //std::cout << depth_camera_ptrs[cam_id]->depth2color_pose()->matrix()<< std::endl;
            depth_pose.matrix() =
                color_camera_ptrs[cam_id]->camera2world_pose().matrix()*depth_camera_ptrs[cam_id]->depth2color_pose()->matrix();
            depth_camera_ptrs[cam_id]->set_camera2world_pose(depth_pose);
          }
        }
      }
      
      
      // Visualize pose update
      int update_save_idx =
          corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
      tracker_ptr->VisualizeOptimization(update_save_idx);
    }
    //for (auto viewer_ptr : tracker_ptr->viewer_ptrs()) viewer_ptr->SetUp();
  }

  // Calculate results
  tracker_ptr->CalculateResults(iteration);

  // Visualize results and update viewers
  tracker_ptr->VisualizeResults(iteration);
}

void ExecuteMeasuredRefinementCycle(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    int iteration) {
  std::chrono::high_resolution_clock::time_point begin_time;
  auto refiner_ptr{tracker_ptr->refiner_ptrs()[0]};

  // Update Cameras
  tracker_ptr->UpdateCameras(iteration);

  std::cout << refiner_ptr->n_corr_iterations() << std :: endl;
  for (int corr_iteration = 0;
       corr_iteration < refiner_ptr->n_corr_iterations(); ++corr_iteration) {
    // Start modalities
    begin_time = std::chrono::high_resolution_clock::now();
    refiner_ptr->StartModalities(corr_iteration);

    // Calculate correspondences
    begin_time = std::chrono::high_resolution_clock::now();
    refiner_ptr->CalculateCorrespondences(corr_iteration);

    // Visualize correspondences
    int corr_save_idx = corr_iteration;
    refiner_ptr->VisualizeCorrespondences(corr_save_idx);

    for (int update_iteration = 0;
         update_iteration < refiner_ptr->n_update_iterations();
         ++update_iteration) {
      // Calculate gradient and hessian
      refiner_ptr->CalculateGradientAndHessian(corr_iteration,
                                               update_iteration);

      // Calculate optimization
      refiner_ptr->CalculateOptimization(corr_iteration, update_iteration);


      // Visualize pose update
      int update_save_idx =
          corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
      refiner_ptr->VisualizeOptimization(update_save_idx);
    }
  }

  // Update viewer
  //tracker_ptr->UpdateViewers(iteration);

}


void SetBodyAndJointPoses(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    const std::map<int, m3t::Transform3fA> body2world_poses_map, std::string obj_name) {
  int idx = 0;
  for (const auto &[body_id, obj_pose] : body2world_poses_map) {
    for (auto &body_ptr : tracker_ptr->body_ptrs()) {
      if (body_ptr->name() == body_names_map[obj_name][body_id])
        body_ptr->set_body2world_pose(obj_pose);
    }
    idx++;
  }
}

Eigen::MatrixXf toEigenMatrix(std::vector<std::vector<float>> matrix) {
  Eigen::MatrixXf outMat(matrix.size(), matrix[0].size());
  for (int i = 0; i < matrix.size(); i++) {
    outMat.row(i) = Eigen::VectorXf::Map(&matrix[i][0], matrix[0].size());
  }
  return outMat;
}

void global_scene_fusion(cosypose::Ransac_candidates &candidates,
                         std::map<int, m3t::Transform3fA> &body2world_poses_map,
                         std::map<int, m3t::Transform3fA> TWC,
                         std::vector<int> inliers_id,
                         std::vector<int> &inlier_labels) {
  std::vector<int> view_ids = candidates.view_ids;
  std::vector<int> obj_ids = candidates.labels;
  torch::Tensor object_pose_tensors = candidates.poses;
  m3t::Transform3fA obj_pose, cam_pose;
  for (int detect_id = 0; detect_id < view_ids.size();
       detect_id++) {
    if (TWC.find(view_ids[detect_id]) != TWC.end()
        //&&std::count(inliers_id.begin(), inliers_id.end(), detect_id)>0
    ) {
      cam_pose = TWC[view_ids[detect_id]];
      Eigen::MatrixXf E = Eigen::Matrix4f::Zero();
      Tensor2EigenMat(object_pose_tensors.index({detect_id}), E);
      obj_pose.matrix() = E;
      body2world_poses_map[obj_ids[detect_id]] = cam_pose * obj_pose;
      inlier_labels.push_back(obj_ids[detect_id]);
    }
  }
}


void Posedetector(mmdeploy_pose_detector_t pose_detector, std::vector<cv::Mat> images, std::vector<cv::Matx33d> cam_Ks,
                  std::map<int, std::vector<int>> &visible_map,
                  cosypose::Ransac_candidates &candidates, int num_id, std::string obj_name) {
  int batch_size = images.size();
  std::vector<mmdeploy_mat_t> mats;

  std::vector<cv::Mat> imgs;
  for (int i = 0; i < batch_size; ++i) {
    cv::Mat img = images[i];
    imgs.push_back(img);
    mmdeploy_mat_t mat{img.data,
                       img.rows,
                       img.cols,
                       3,
                       MMDEPLOY_PIXEL_FORMAT_BGR,
                       MMDEPLOY_DATA_TYPE_UINT8};
    mats.push_back(mat);
  }
  mmdeploy_pose_detection_t *res{};
  int status = mmdeploy_pose_detector_apply(pose_detector, mats.data(), batch_size,
                                        &res);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply pose estimator, code: %d\n", (int)status);
  }
  mmdeploy_pose_detection_t *res_ptr = res;
  int num_point = 8;
  std::vector<cv::Point2f> kpts_2d;
  m3t::Transform3fA body_pose;
  memset(&body_pose, 0, sizeof(body_pose));
  
  std::vector<int> view_ids, labels;
  std::vector<m3t::Transform3fA> body2world_poses;
  for (int img_i = 0; img_i < (int)mats.size(); ++img_i) {
    for (int i = 0; i < res_ptr->num_id; i++) {
      int obj_id = res_ptr->label_id[i];
      float score = res_ptr->score[i];
      
      //std::cout << obj_id << std::endl;
      kpts_2d.clear();
      for (int k = 0; k < num_point; k++) {
        kpts_2d.push_back(
            cv::Point2f((int)res_ptr->point[i * num_point + k].x,
                        (int)res_ptr->point[i * num_point + k].y));
      }
      cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
      cv::Mat rvec, tvec;

      bool solution;
      solution =
          cv::solvePnPRansac(kpts_gt_map[obj_name][obj_id], kpts_2d,
                             cam_Ks[img_i], distCoeffs,
                                    rvec, tvec, false);
      
      cv::Mat R;
      cv::Rodrigues(rvec, R);  // R is 3x3
      cv::Mat T(4, 4, R.type());
      T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1;     // copies R into T
      T(cv::Range(0, 3), cv::Range(3, 4)) = tvec * 1;  // copies tvec into T
      Eigen::Matrix4f eigen_mat;
      cv::cv2eigen(T, eigen_mat);
      eigen_mat.row(3) << 0, 0, 0, 1;
      m3t::Transform3fA body2world_pose;
      body2world_pose.matrix() = eigen_mat;
      if (body2world_pose.translation()(0) != 0 &&
          body2world_pose.translation()(1) != 0 &&
          body2world_pose.translation()(2) != 0) {
        visible_map[img_i].push_back(obj_id);
        view_ids.push_back(img_i);
        labels.push_back(obj_id);
        body2world_poses.push_back(body2world_pose);
      }
      
    }
    res_ptr = res_ptr + 1;
  }
  int detect_num = body2world_poses.size();
  torch::Tensor poses = torch::zeros({detect_num, 4, 4});  // detect_id, 4, 4
  for (int id = 0; id < detect_num; id++) {
    auto vec = body2world_poses[id].matrix();
    torch::Tensor T;
    Eigen::MatrixXf mat = vec;
    EigenMat2Tensor(T, mat);
    poses.index({id}) = T;
  }
  candidates.labels = labels;
  candidates.view_ids = view_ids;
  candidates.poses = poses;
  
}

bool tryParse(std::string& input, int& output) {
    try {
        output = std::stoi(input);
    }
    catch (std::invalid_argument) {
        return false;
    }
    return true;
}

int ProcessKeyboardInput() {
  std::string keyboardChar;

  while (true) {
    keyboardChar = getch();
    // switch object detection models

    /*
    if (keyboardChar == "o") {
        for (int obj_id = 0; obj_id < obj_list.size(); ++obj_id) {
            std::cout << std::to_string(obj_id) + ": " << obj_list[obj_id]
                << '\n';
        }
        std::cout << "Select the object with the number: ";
        std::string input;
        getline(std::cin, input);

        int obj_id;
        while (!tryParse(input, obj_id)) {
            std::cout << "Bad entry. Enter a NUMBER: ";
            getline(std::cin, input);
        }
        object_name = obj_list[obj_id];
        
    }*/

    if (keyboardChar == "b") flag = 1;
    auto begin = std::chrono::high_resolution_clock::now();
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    while (true) {
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      if ((elapsed > 500) || (flag_send == 1)) break;
    }
    flag = 0;
  }
}

int main(int argc, char *argv[]) {
  
  // Load YOLOpose
  std::unordered_map<std::string, mmdeploy_pose_detector_t> pose_detector_map;
  for (auto iter : body_names_map) {
     std::string obj_name = iter.first;

     for (int obj_id = 0; obj_id < body_names_map[obj_name].size(); obj_id++)
       body_name2idx_map_all[obj_name].insert(
           {body_names_map[obj_name][obj_id], obj_id});
     for (const auto &[body_name, idx] : body_name2idx_map_all[obj_name]) {
       std::cout << body_name;
       std::cout << idx << std::endl;

       // set model directory
       


       // read model infos

       cv::FileStorage fs_model;

       std::filesystem::path model_info_path{ data_root_dir + obj_name + "/models/models_info.json" };

       try {
         fs_model.open(cv::samples::findFile(model_info_path.string()),
                       cv::FileStorage::READ);
       } catch (cv::Exception e) {
       }
       if (!fs_model.isOpened()) {
         std::cerr << "Could not open file " << model_info_path << std::endl;
         return false;
       }

       symmetry_ids_map[obj_name].clear();
       symmetries_map[obj_name].clear();
       symmetries_map[obj_name].resize(
           body_name2idx_map_all[obj_name].size());

       for (const auto &[body_name, idx] : body_name2idx_map_all[obj_name]) {
         // std::cout << body_name << std::endl;
         // read object model infos
         cv::FileNode fn_object{fs_model[std::to_string(idx)]};
         cv::FileNode fn_discrete{fn_object["symmetries_discrete"]};
         cv::FileNode fn_continuous{fn_object["symmetries_continuous"]};
         if (!fn_continuous.empty() || !fn_discrete.empty())
           symmetry_ids_map[obj_name].push_back(idx);

         std::vector<m3t::Transform3fA> all_M_discrete, all_M_continuous, all_M;
         m3t::Transform3fA sym_matrix;
         sym_matrix.matrix() = Eigen::Matrix4f::Identity();
         all_M_discrete.push_back(sym_matrix);
         // read discrete symmetry infomation
         if (!fn_discrete.empty()) {
           for (int sym_idx = 0; sym_idx < fn_discrete.size(); sym_idx++) {
             cv::FileNode fn_matrix{fn_discrete[sym_idx]};
             if (fn_matrix.empty()) break;

             for (int i = 0; i < 16; ++i)
               fn_matrix[i] >> sym_matrix.matrix()(i / 4, i % 4);
             sym_matrix.translation() *= 0.001;
             // std::cout << sym_matrix.matrix() << std::endl;
             // symmetries_[idx-1].push_back(sym_matrix);
             all_M_discrete.push_back(sym_matrix);
           }
         }

         // read continuous symmetry information
         if (!fn_continuous.empty()) {
           for (int sym_axis = 0; sym_axis < fn_continuous.size(); sym_axis++) {
             cv::FileNode fn_axis{fn_continuous[sym_axis]};
             float axis1, axis2, axis3, offset1, offset2, offset3;
             axis1 = fn_axis["axis"][0];
             axis2 = fn_axis["axis"][1];
             axis3 = fn_axis["axis"][2];
             sym_matrix.translation()[0] = fn_axis["offset"][0];
             sym_matrix.translation()[1] = fn_axis["offset"][1];
             sym_matrix.translation()[2] = fn_axis["offset"][2];
             Eigen::Matrix3f m;
             for (int sym_idx = 0; sym_idx < n_sym; sym_idx++) {
               m = Eigen::AngleAxisf(2 * M_PI * axis1 * sym_idx / n_sym,
                                     Eigen::Vector3f::UnitX()) *
                   Eigen::AngleAxisf(2 * M_PI * axis2 * sym_idx / n_sym,
                                     Eigen::Vector3f::UnitY()) *
                   Eigen::AngleAxisf(2 * M_PI * axis3 * sym_idx / n_sym,
                                     Eigen::Vector3f::UnitZ());

               sym_matrix.matrix().block(0, 0, 3, 3) = m;
               sym_matrix.translation() *= 0.001;
               // std::cout << sym_matrix.matrix() << std::endl;
               // symmetries_[idx - 1].push_back(sym_matrix);
               all_M_continuous.push_back(sym_matrix);
             }
           }
         }
         for (auto sym_d : all_M_discrete) {
           if (all_M_continuous.size() > 0) {
             for (auto sym_c : all_M_continuous) {
               auto M = sym_c * sym_d;
               all_M.push_back(M);
             }

           } else {
             all_M.push_back(sym_d);
           }
         }
         symmetries_map[obj_name][idx] = all_M;
       }

       // read key points data
       std::ifstream accfile;
       std::string model_file_path = data_root_dir + obj_name + "/kpts";
       std::string path_kpts = model_file_path + "/" + body_name + ".txt";
       std::cout << path_kpts << std::endl;
       accfile.open(path_kpts);
       std::vector<cv::Point3f> numbers;

       if (!accfile.is_open()) {
         std::cout << "open key point file failed" << std::endl;
       }

       for (int i = 0; !accfile.eof(); i++) {
         double x = -1, y = -1, z = -1;
         accfile >> x >> y >> z;
         if (x > -1) {
           numbers.push_back(cv::Point3f(x, y, z));
           // std::cout << x << std::endl;
         }
       }
       if (accfile.is_open()) {
         accfile.close();
       }

       kpts_gt_map[obj_name].push_back(numbers);
       // std::cout << kpts_gt[0].size() << std::endl;
     }
     // prepare mesh database
     int num_obj = kpts_gt_map[obj_name].size();
     int num_kpts = kpts_gt_map[obj_name][0].size();
     kpts_3d = torch::zeros({num_obj, num_kpts, 3});
     for (int obj_id = 0; obj_id < num_obj; obj_id++) {
       for (int kpts_id = 0; kpts_id < num_kpts; kpts_id++) {
         kpts_3d.index({obj_id, kpts_id, 0}) =
             kpts_gt_map[obj_name][obj_id][kpts_id].x;
         kpts_3d.index({obj_id, kpts_id, 1}) =
             kpts_gt_map[obj_name][obj_id][kpts_id].y;
         kpts_3d.index({obj_id, kpts_id, 2}) =
             kpts_gt_map[obj_name][obj_id][kpts_id].z;
       }
     }
     sym_tensor = pad_stack_tensors(symmetries_map[obj_name]);
     for (int obj_id = 0; obj_id < num_obj; obj_id++) {
       n_sym_mapping_all[obj_name][obj_id] =
           symmetries_map[obj_name][obj_id].size();
     }
     mesh_db_map[obj_name].kpts_3d = kpts_3d;
     mesh_db_map[obj_name].n_sym_mapping = n_sym_mapping_all[obj_name];
     mesh_db_map[obj_name].sym_tensor = sym_tensor;

     mmdeploy_pose_detector_t pose_detector{};
     int status{};

     // Load pose detector
     std::filesystem::path model_path{ model_root_dir + "checkpoint_" + obj_name };
     status = mmdeploy_pose_detector_create_by_path(
         model_path.generic_string().c_str(), device_name, 0, &pose_detector);
     if (status != MMDEPLOY_SUCCESS) {
       fprintf(stderr, "failed to create pose_estimator, code: %d\n",
               (int)status);
       return 1;
     }

     pose_detector_map[obj_name] = pose_detector;
  }
  

  
  
  
  std::vector<mmdeploy_mat_t> mats;
  cv::Mat img = cv::Mat::ones(1280, 720, CV_8UC3);
  mmdeploy_mat_t mat{img.data,
                     img.rows,
                     img.cols,
                     3,
                     MMDEPLOY_PIXEL_FORMAT_BGR,
                     MMDEPLOY_DATA_TYPE_UINT8};
  mats.push_back(mat);
  // warmup
  //for (int i = 0; i < 20; ++i) {
  //  mmdeploy_pose_detection_t *res{};
  //  status = mmdeploy_pose_detector_apply(pose_detector, mats.data(),
  //                                        1, &res);
  //  mmdeploy_pose_detector_release_result(res, mats.size());
  //}
  
  constexpr bool kUseDepthViewer = false;
  constexpr bool kUseRegionModality = true;
  constexpr bool kUseTextureModality = false;
  constexpr bool kUseDepthModality = false;
  constexpr bool kUseHololensdepth = false;
  constexpr bool kMeasureOcclusions = true;
  constexpr bool kModelOcclusions = false;
  constexpr bool kVisualizePoseResult = false;
  constexpr bool kSaveImages = false;
  const std::filesystem::path save_directory{""};

  // Set up tracker and renderer geometry
  auto tracker_ptr{std::make_shared<m3t::Tracker>("tracker")};
  auto renderer_geometry_ptr{
      std::make_shared<m3t::RendererGeometry>("renderer geometry")};

  // Set up cameras
  std::cout << "Please provide HOLOLENS IP address: ";
  std::string input;
  getline(std::cin, input);
  // char const *host = "192.168.0.149";
  char const *host = input.c_str();

  std::thread press_button(ProcessKeyboardInput);
  press_button.detach();
  std::vector<std::shared_ptr<m3t::AzureKinectColorMulti>> color_camera_ptrs{};
  std::vector<std::shared_ptr<m3t::AzureKinectDepthMulti>> depth_camera_ptrs{};
  std::vector<std::shared_ptr<m3t::FocusedBasicDepthRenderer>> color_depth_renderer_ptrs{};
  std::vector<std::shared_ptr<m3t::FocusedBasicDepthRenderer>> depth_depth_renderer_ptrs{};
  std::vector<std::shared_ptr<m3t::FocusedSilhouetteRenderer>> color_silhouette_renderer_ptrs{};
  std::vector<std::shared_ptr<m3t::Body>> body_ptrs{};
  std::vector<std::shared_ptr<m3t::Link>> link_ptrs{};
  //std::vector<std::vector<std::vector<std::shared_ptr<m3t::Modality>>>>
  //    cam_link_ptrs{};
  //cam_link_ptrs.resize(num_cam);
  
  client = hl2ss::lnm::ipc_umq(host, hl2ss::ipc_port::UNITY_MESSAGE_QUEUE);
  auto hololens_color_ptr{
      std::make_shared<m3t::HololensColorCamera>(host, "hololens_color")};
  
  //auto hololens_depth_ptr{std::make_shared<m3t::HololensDepthCamera>(host, "hololens_depth")};
  auto color_viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
      "color_viewer_hololens", hololens_color_ptr,
      renderer_geometry_ptr)};
  if (kSaveImages) color_viewer_ptr->StartSavingImages(save_directory);
  tracker_ptr->AddViewer(color_viewer_ptr);
  
  /*
  if (kUseHololensdepth) {
    auto depth_viewer_ptr{std::make_shared<m3t::NormalDepthViewer>(
        "depth_viewer_hololens", hololens_depth_ptr, renderer_geometry_ptr,
        0.3f, 1.0f)};
    if (kSaveImages) depth_viewer_ptr->StartSavingImages(save_directory);
    tracker_ptr->AddViewer(depth_viewer_ptr);
  }*/
  
  // Set up depth renderer
  auto color_depth_renderer_hololens_ptr{
      std::make_shared<m3t::FocusedBasicDepthRenderer>(
          "color_depth_renderer_hololens", renderer_geometry_ptr,
          hololens_color_ptr)};
  
  /*
  auto depth_depth_renderer_hololens_ptr{
      std::make_shared<m3t::FocusedBasicDepthRenderer>(
          "depth_depth_renderer_hololens", renderer_geometry_ptr,
          hololens_depth_ptr)};*/
  // Set up silhouette renderer
  auto color_silhouette_renderer_hololens_ptr{
      std::make_shared<m3t::FocusedSilhouetteRenderer>(
          "color_silhouette_renderer_hololens", renderer_geometry_ptr,
          hololens_color_ptr)};
  
  
  for (int cam_id = 0; cam_id < num_cam; ++cam_id) {
    //cam_link_ptrs[cam_id].resize(body_names.size());
    auto color_camera_ptr{std::make_shared<m3t::AzureKinectColorMulti>(
        cam_id, "azure_kinect_color" + std::to_string(cam_id))};
    auto depth_camera_ptr{std::make_shared<m3t::AzureKinectDepthMulti>(
        cam_id, "azure_kinect_depth" + std::to_string(cam_id))};
    color_camera_ptrs.push_back(color_camera_ptr);
    depth_camera_ptrs.push_back(depth_camera_ptr);

    // Set up viewers
    auto color_viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
        "color_viewer" + std::to_string(cam_id), color_camera_ptr,
        renderer_geometry_ptr)};
    if (kSaveImages) color_viewer_ptr->StartSavingImages(save_directory);
    tracker_ptr->AddViewer(color_viewer_ptr);

    if (kUseDepthViewer) {
    auto depth_viewer_ptr{std::make_shared<m3t::NormalDepthViewer>(
          "depth_viewer" + std::to_string(cam_id), depth_camera_ptr,
          renderer_geometry_ptr, 0.3f, 1.0f)};
    if (kSaveImages)
    depth_viewer_ptr->StartSavingImages(save_directory);
    tracker_ptr->AddViewer(depth_viewer_ptr);
    }
    // Set up depth renderer
    auto color_depth_renderer_ptr{
        std::make_shared<m3t::FocusedBasicDepthRenderer>(
            "color_depth_renderer" + std::to_string(cam_id),
            renderer_geometry_ptr, color_camera_ptr)};
    color_depth_renderer_ptrs.push_back(color_depth_renderer_ptr);
    auto depth_depth_renderer_ptr{
        std::make_shared<m3t::FocusedBasicDepthRenderer>(
            "depth_depth_renderer" + std::to_string(cam_id),
            renderer_geometry_ptr, depth_camera_ptr)};
    depth_depth_renderer_ptrs.push_back(depth_depth_renderer_ptr);
    // Set up silhouette renderer
    auto color_silhouette_renderer_ptr{
        std::make_shared<m3t::FocusedSilhouetteRenderer>(
            "color_silhouette_renderer" + std::to_string(cam_id),
            renderer_geometry_ptr,
            color_camera_ptr)};
    color_silhouette_renderer_ptrs.push_back(color_silhouette_renderer_ptr);
  }

  for (auto iter : body_names_map) {
    std::string obj_name = iter.first;
    int obj_id = 0;
    // auto refiner_ptr{std::make_shared<m3t::Refiner>("refiner", 20)};
    for (const auto body_name : body_names_map[obj_name]) {
      // Set up body
      std::filesystem::path objfile_path{data_root_dir + obj_name + "/models/" + (body_name + ".obj")};
      m3t::Transform3fA model_ori{Eigen::Matrix4f::Identity()};
      // obj_pose.matrix() = Eigen::Matrix4f::Identity();
      float geometry_unit_in_meter = 0.001;
      auto body_ptr{std::make_shared<m3t::Body>(
          body_name, objfile_path, geometry_unit_in_meter, 1, 1, model_ori)};
      // std::cout << "region_id: ";
      // std::cout << static_cast<int>(body_ptr->get_id(m3t::IDType::REGION)) <<
      // std::endl;
      body_ptrs.push_back(body_ptr);
      renderer_geometry_ptr->AddBody(body_ptr);
      color_depth_renderer_hololens_ptr->AddReferencedBody(body_ptr);
      // depth_depth_renderer_hololens_ptr->AddReferencedBody(body_ptr);
      color_silhouette_renderer_hololens_ptr->AddReferencedBody(body_ptr);
      for (auto color_depth_renderer_ptr : color_depth_renderer_ptrs)
        color_depth_renderer_ptr->AddReferencedBody(body_ptr);
      for (auto depth_depth_renderer_ptr : depth_depth_renderer_ptrs)
        depth_depth_renderer_ptr->AddReferencedBody(body_ptr);
      for (auto color_silhouette_renderer_ptr : color_silhouette_renderer_ptrs)
        color_silhouette_renderer_ptr->AddReferencedBody(body_ptr);

      // Set up link
      auto link_ptr{std::make_shared<m3t::Link>(body_name + "_link", body_ptr)};
      link_ptrs.push_back(link_ptr);

      obj_id++;
    }
  }

  std::cout << "obj model loaded!!!" << std::endl;
  //tracker_ptr->AddRefiner(refiner_ptr);

  // Start tracking
  if (!tracker_ptr->SetUp()) return -1;
  client->open();
  //if (!tracker_ptr->RunTrackerProcess(true, false)) return -1;
  std::vector<std::vector<m3t::Transform3fA>> body2world_poses_all;
  std::vector<m3t::Transform3fA> body2world_poses, body_pose_final;
  
  std::shared_ptr<std::vector<m3t::Transform3fA>> body_pose_ptr =
      std::make_shared<std::vector<m3t::Transform3fA>>(body_pose_final);
  Eigen::Matrix4f Trans;
  std::vector<cv::Mat> images;
  std::vector<cv::Matx33d> cam_Ks;
  std::vector<m3t::Transform3fA> cam2worlds(num_cam);
  
  //std::thread tViewer = std::thread(DrawTrajectory, hololens_color_ptr, color_camera_ptrs, body_pose_ptr);
  //tViewer.detach();

  std::thread publisher = std::thread(UpdatePublisher);
  publisher.detach();
  std::cout << "demo starts" << std::endl;
  std::map<int, m3t::Transform3fA> TWC;
  std::map<int, m3t::Transform3fA> body2world_poses_map;
  for (int iteration = 0;; ++iteration) {
    //std::cout << "Update camera" << std::endl;
    std::string obj_name = object_name;
    if (!tracker_ptr->UpdateCameras(iteration)) return false;
    
    images.clear();
    cam_Ks.clear();

    cv::Mat image = hololens_color_ptr->image();
    images.push_back(image);
    cv::Matx33d cam_K = {hololens_color_ptr->intrinsics().fu,
                         0.0,
                         hololens_color_ptr->intrinsics().ppu,
                         0.0,
                         hololens_color_ptr->intrinsics().fv,
                         hololens_color_ptr->intrinsics().ppv,
                         0.0,
                         0.0,
                         1.0};
    cam_Ks.push_back(cam_K);
    
    for (auto camera_ptr : color_camera_ptrs)
    {
        cv::Mat image = camera_ptr->image();
        images.push_back(image);
        cv::Matx33d cam_K = {camera_ptr->intrinsics().fu,
               0.0,
               camera_ptr->intrinsics().ppu,
               0.0,
               camera_ptr->intrinsics().fv,
               camera_ptr->intrinsics().ppv,
               0.0,
               0.0,
               1.0};
        cam_Ks.push_back(cam_K);
        
    }

    //std::cout << "Pose detection" << std::endl;
    cosypose::Ransac_candidates candidates;
    std::map<int, std::vector<int>> visible_map;
    Posedetector(pose_detector_map[obj_name], images, cam_Ks, visible_map,
                 candidates,
                 body_names_map[obj_name].size(), obj_name);
    //std::cout << "Pose detection done" << std::endl;
    cosypose::Results cam_poses_results;
    //std::cout << "candidates matching begin" << std::endl;
    int model_bsz = 1e3;
    int score_bsz = 1e5;
    float dist_threshold = 0.05;
    int n_ransac_iter = 2000;
    int n_min_inliers = 3;
    std::vector<int> inlier_ids;
    cosypose::multiview_candidate_matching(
        candidates, mesh_db_map[obj_name], cam_poses_results, inlier_ids,
        model_bsz,
        score_bsz, dist_threshold, n_ransac_iter, n_min_inliers);

    //std::cout << "candidates matching done" << std::endl;
    using ViewPair = std::tuple<int, int>;

    std::map<ViewPair, m3t::Transform3fA> T_camera_map;
    for (int cam_idx = 0; cam_idx < cam_poses_results.view1.size(); cam_idx++) {
      int v1 = cam_poses_results.view1[cam_idx];
      int v2 = cam_poses_results.view2[cam_idx];
      // std::cout << v1 << v2 << std::endl;
      ViewPair view_pair_C1C2(v1, v2);
      ViewPair view_pair_C2C1(v2, v1);
      torch::Tensor pose_tensor = cam_poses_results.TC1C2.index({cam_idx});
      m3t::Transform3fA pose;
      Eigen::MatrixXf E = Eigen::Matrix4f::Zero();
      Tensor2EigenMat(pose_tensor, E);
      pose.matrix() = E;
      T_camera_map[view_pair_C1C2] = pose;
      T_camera_map[view_pair_C2C1] = pose.inverse();
    }

    std::vector<ViewPair> keys;
    keys.reserve(T_camera_map.size());
    for (const auto &[key, value] : T_camera_map) {
      keys.push_back(key);
    }
    // Initialize object poses in Hololens view
    // Initialize views
    int n_pass = 20;
    int n = 0;
    
    m3t::Transform3fA camera2world_pose{hololens_color_ptr->camera2world_pose()};
    //m3t::Transform3fA camera2world_pose{m3t::Transform3fA::Identity()};
    
    TWC[0] = camera2world_pose;  // initialize main camera with hololens pose
    std::vector<int> views_initialized = {0};
    std::vector<int> views_to_initialize(num_cam),
        views_ordered(num_cam + 1);
    std::iota(views_to_initialize.begin(), views_to_initialize.end(), 1);
    std::iota(views_ordered.begin(), views_ordered.end(), 0);
    while (views_to_initialize.size() > 0) {
      for (auto v1 : views_ordered) {
        if (std::count(views_to_initialize.begin(), views_to_initialize.end(),
                       v1) > 0) {
          for (auto v2 : views_ordered) {
            if (std::count(views_initialized.begin(), views_initialized.end(),
                           v2) < 1)
              continue;

            ViewPair view_pair(v2, v1);

            if (std::count(keys.begin(), keys.end(), view_pair) > 0) {
              m3t::Transform3fA TC2C1 = T_camera_map[view_pair];
              m3t::Transform3fA TWC2 = TWC[v2];
              TWC[v1] = TWC2 * TC2C1;
              views_to_initialize.erase(std::find(
                  views_to_initialize.begin(), views_to_initialize.end(), v1));
              color_camera_ptrs[v1 - 1]->set_camera2world_pose(TWC[v1]);

              m3t::Transform3fA depth2color;
              depth2color.matrix() = depth_camera_ptrs[v1 - 1]->depth2color_pose()->matrix();
              depth_camera_ptrs[v1 - 1]->set_camera2world_pose(TWC[v1] *
                                                               depth2color);
              views_initialized.push_back(v1);
              break;
            }
          }
        }
      }
      n += 1;
      if (n >= n_pass) {
        // std::cerr << "Cannot find an initialization" <<std::endl;
        break;
      }
    }
    // Fuse object poses from different views
    
    std::vector<int> inlier_labels;
    global_scene_fusion(candidates, body2world_poses_map, TWC, inlier_ids,
                        inlier_labels);
    
    
    SetBodyAndJointPoses(tracker_ptr, body2world_poses_map, obj_name);
    //std::cout << "Update tracker" << std::endl;
    if (!tracker_ptr->UpdateSubscribers(iteration)) return false;
    if (!tracker_ptr->CalculateConsistentPoses()) return false;
    //if (!tracker_ptr->ExecuteDetectingStep(iteration)) return false;
    //std::cout << "BA" << std::endl;
    if (!tracker_ptr->ExecuteStartingStep(iteration)) return false;
    //ExecuteTracking(tracker_ptr, color_camera_ptrs, depth_camera_ptrs,link_ptrs, cam_link_ptrs, iteration);
    //ExecuteMeasuredRefinementCycle(tracker_ptr, iteration);
    for (auto viewer_ptr : tracker_ptr->viewer_ptrs()) viewer_ptr->SetUp();
    if (!tracker_ptr->UpdateViewers(iteration)) return false;
    //if (iteration % 5 == 0) {
    //    if (!UpdatePublisher(tracker_ptr, obj_name)) return false;
    //}
    
    //std::cout << "Finish" << std::endl;
    
    // sum up pose information
    json = "[";
    for (const auto &[body_name, idx] : body_name2idx_map_all[obj_name]) {
      const auto &body_ptr_ = *std::find_if(
          begin(tracker_ptr->body_ptrs()), end(tracker_ptr->body_ptrs()),
          [&](const auto &b) { return b->name() == body_name; });
      float x = body_ptr_->body2world_pose().translation().x();
      std::string trans1_x = std::to_string(x);
      float y = -body_ptr_->body2world_pose().translation().y();
      std::string trans1_y = std::to_string(y);
      float z = body_ptr_->body2world_pose().translation().z();
      std::string trans1_z = std::to_string(z);
      Eigen::Matrix3f rot = body_ptr_->world2body_pose().rotation();
      Eigen::Vector3f eulerAngle = rot.eulerAngles(2, 0, 1);
      float qx = eulerAngle[1] * 180 / 3.14159;
      std::string trans1_qx = std::to_string(qx);
      float qy = -eulerAngle[2] * 180 / 3.14159;
      std::string trans1_qy = std::to_string(qy);
      float qz = eulerAngle[0] * 180 / 3.14159 + 180;
      std::string trans1_qz = std::to_string(qz);
      std::string name = body_name;
      if (strstr(body_name.c_str(), "Copy") != NULL) {
        int index_Copy = body_name.find("Copy", 0);
        int index_num = std::stoi(body_name.substr(
            index_Copy + 4, body_name.length() - index_Copy - 4));
        name = body_name.substr(0, index_Copy + 4) + std::to_string(index_num);
      }
      if (z != 0) {
        json += "{  \"classId\": \"" + name + "\" , \"x\": " + trans1_x +
                "\ , \"y\": " + trans1_y + "\ , \"z\": " + trans1_z +
                "\ , \"qx\": " + trans1_qx + "\ , \"qy\": " + trans1_qy +
                "\ , \"qz\": " + trans1_qz + "\},";
      }
    }


  }
  return 0;

}
