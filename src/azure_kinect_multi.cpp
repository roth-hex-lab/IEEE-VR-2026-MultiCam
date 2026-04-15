// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include <m3t/azure_kinect_multi.h>

namespace m3t {


AzureKinectMulti &AzureKinectMulti::GetInstance() {
    static AzureKinectMulti kinect;
  return kinect;
}

AzureKinectMulti::~AzureKinectMulti() {
  if (initial_set_up_) {
    for (size_t i = 0; i < devices_.size(); ++i) {
      devices_[i].stop_cameras();
      devices_[i].close();
    }
    
  }
}

void AzureKinectMulti::UseColorCamera() { use_color_camera_ = true; }

void AzureKinectMulti::UseDepthCamera() { use_depth_camera_ = true; }

int AzureKinectMulti::RegisterID(uint32_t device_index) {
  const std::lock_guard<std::mutex> lock{mutex_};
  // Check if camera is available
  num_devices_ = k4a::device::get_installed_count();
  if (num_devices_ == 0) return false;
  if (update_capture_ids_
      .size() < num_devices_){
      update_capture_ids_.resize(num_devices_);
      next_ids_.resize(num_devices_);
  }

  update_capture_ids_[device_index].insert(
      std::pair<int, bool>{next_ids_[device_index], true});

  return next_ids_[device_index]++;
}


bool AzureKinectMulti::UnregisterID(uint32_t device_index, int id) {
  const std::lock_guard<std::mutex> lock{mutex_};
  return update_capture_ids_[device_index].erase(id);
}

bool AzureKinectMulti::SetUp() {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!initial_set_up_) {
    // Configure camera
    config_ = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config_.camera_fps = K4A_FRAMES_PER_SECOND_15;
    config_.synchronized_images_only = true;
    //if (device_index == 0) 
    //  config_.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
    //else
     // config_.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
    if (use_color_camera_) {
      config_.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
      config_.color_resolution = K4A_COLOR_RESOLUTION_720P;
    }
    if (use_depth_camera_) {
      config_.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    }


    // Start camera
    //device_ = k4a::device::open(K4A_DEVICE_DEFAULT);
    captures_.resize(num_devices_);
    devices_.resize(num_devices_);
    calibrations_.resize(num_devices_);
    color2depth_poses_.resize(num_devices_);
    depth2color_poses_.resize(num_devices_);
    for (size_t device_id = 0; device_id < num_devices_; ++device_id) {
      devices_[device_id] = k4a::device::open(device_id);
      devices_[device_id].start_cameras(&config_);

             // Get calibration
      calibrations_[device_id] = devices_[device_id].get_calibration(
             config_.depth_mode,
                                                config_.color_resolution);

         // Get extrinsics and calculate pose
         if (use_color_camera_ && use_depth_camera_) {
           const k4a_calibration_extrinsics_t extrinsics{
               devices_[device_id]
                   .get_calibration(config_.depth_mode,
                                    config_.color_resolution)
                   .extrinsics[K4A_CALIBRATION_TYPE_COLOR]
                              [K4A_CALIBRATION_TYPE_DEPTH]};
           Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot{extrinsics.rotation};
           Eigen::Vector3f trans{extrinsics.translation};
           color2depth_poses_[device_id].setIdentity();
           color2depth_poses_[device_id].translate(trans * 0.001f);
           color2depth_poses_[device_id].rotate(rot);
           depth2color_poses_[device_id] =
               color2depth_poses_[device_id].inverse();
         }
         
         // Load multiple images to adjust to white balance
         constexpr int kTimeoutInMs = 100;
         constexpr int kNumberImagesDropped = 10;
         
         for (int i = 0; i < kNumberImagesDropped; ++i) {
           while (!devices_[device_id].get_capture(
               &captures_[device_id],
                                       std::chrono::milliseconds{kTimeoutInMs}));
         }
         
    }
    
    

    initial_set_up_ = true;
  }
  return true;
}

bool AzureKinectMulti::UpdateCapture(uint32_t device_index, int id,
                                     bool synchronized) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!initial_set_up_) return false;

  if (update_capture_ids_[device_index].at(id)) {
    if (synchronized)
      devices_[device_index].get_capture(&captures_[device_index],
                                         std::chrono::milliseconds{-1});
    else
      devices_[device_index].get_capture(&captures_[device_index],
                                          std::chrono::milliseconds{0});
    for (auto &[_, v] : update_capture_ids_[device_index]) v = false;
  }
  update_capture_ids_[device_index].at(id) = true;
  return true;
}

bool AzureKinectMulti::use_color_camera() const { return use_color_camera_; }

bool AzureKinectMulti::use_depth_camera() const { return use_depth_camera_; }

const k4a::capture &AzureKinectMulti::capture(uint32_t device_index) const {
  return captures_[device_index];
}

const k4a::calibration &AzureKinectMulti::calibration(
    uint32_t device_index) const {
  return calibrations_[device_index];
}

const Transform3fA *AzureKinectMulti::color2depth_pose(
    uint32_t device_index) const {
  if (initial_set_up_)
    return &color2depth_poses_[device_index];
  else
    return nullptr;
}

const Transform3fA *AzureKinectMulti::depth2color_pose(
    uint32_t device_index) const {
  if (initial_set_up_)
    return &depth2color_poses_[device_index];
  else
    return nullptr;
}

AzureKinectColorMulti::AzureKinectColorMulti(uint32_t device_index,
                                             const std::string &name,
                                               float image_scale,
                                               bool use_depth_as_world_frame)
    : 
      device_index_{device_index},
      ColorCamera{name},
      image_scale_{image_scale},
      use_depth_as_world_frame_{use_depth_as_world_frame},
      azure_kinect_{AzureKinectMulti::GetInstance()} {
  azure_kinect_.UseColorCamera();
  azure_kinect_id_ = azure_kinect_.RegisterID(device_index_);
}

AzureKinectColorMulti::AzureKinectColorMulti(
    uint32_t device_index, const std::string &name, const std::filesystem::path &metafile_path)
    : ColorCamera{name, metafile_path},
      azure_kinect_{AzureKinectMulti::GetInstance()} {
  azure_kinect_.UseColorCamera();
  azure_kinect_id_ = azure_kinect_.RegisterID(device_index_);
}

AzureKinectColorMulti::~AzureKinectColorMulti() {
  azure_kinect_.UnregisterID(device_index_, azure_kinect_id_);
}

bool AzureKinectColorMulti::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;
  if (!initial_set_up_ && !azure_kinect_.SetUp())
    return false;
  if (use_depth_as_world_frame_)
    set_camera2world_pose(*azure_kinect_.color2depth_pose(device_index_));
  GetIntrinsicsAndDistortionMap();
  SaveMetaDataIfDesired();
  set_up_ = true;
  initial_set_up_ = true;
  
  return UpdateImage(true);
}

void AzureKinectColorMulti::set_image_scale(float image_scale) {
  image_scale_ = image_scale;
  set_up_ = false;
}

void AzureKinectColorMulti::set_use_depth_as_world_frame(
    bool use_depth_as_world_frame) {
  use_depth_as_world_frame_ = use_depth_as_world_frame;
  set_up_ = false;
}

bool AzureKinectColorMulti::UpdateImage(bool synchronized) {
  if (!set_up_) {
    std::cerr << "Set up azure kinect color camera " << name_ << " first"
              << std::endl;
    return false;
  }
  
  // Get and undistort image
  cv::Mat temp_image;
  azure_kinect_.UpdateCapture(device_index_, azure_kinect_id_, synchronized);
  cv::cvtColor(
      cv::Mat{cv::Size{intrinsics_.width, intrinsics_.height}, CV_8UC4,
                       (void *)azure_kinect_.capture(device_index_)
                           .get_color_image()
                           .get_buffer(),
              cv::Mat::AUTO_STEP},
      temp_image, cv::COLOR_RGBA2RGB);
  cv::remap(temp_image, image_, distortion_map_, cv::Mat{}, cv::INTER_NEAREST,
            cv::BORDER_CONSTANT);

  SaveImageIfDesired();
  return true;
}

float AzureKinectColorMulti::image_scale() const { return image_scale_; }

bool AzureKinectColorMulti::use_depth_as_world_frame() const {
  return use_depth_as_world_frame_;
}

const Transform3fA *AzureKinectColorMulti::color2depth_pose() const {
  return azure_kinect_.color2depth_pose(device_index_);
}

const Transform3fA *AzureKinectColorMulti::depth2color_pose() const {
  return azure_kinect_.depth2color_pose(device_index_);
}

bool AzureKinectColorMulti::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  ReadOptionalValueFromYaml(fs, "camera2world_pose", &camera2world_pose_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_index", &save_index_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
  ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
  ReadOptionalValueFromYaml(fs, "image_scale", &image_scale_);
  ReadOptionalValueFromYaml(fs, "use_depth_as_world_frame",
                            &use_depth_as_world_frame_);
  fs.release();

  // Process parameters
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  world2camera_pose_ = camera2world_pose_.inverse();
  return true;
}

void AzureKinectColorMulti::GetIntrinsicsAndDistortionMap() {
  // Load intrinsics from camera
  const k4a_calibration_camera_t calibration{
      azure_kinect_.calibration(device_index_).color_camera_calibration};
  const k4a_calibration_intrinsic_parameters_t::_param param =
      calibration.intrinsics.parameters.param;
  intrinsics_.fu = param.fx;
  intrinsics_.fv = param.fy;
  intrinsics_.ppu = param.cx;
  intrinsics_.ppv = param.cy;
  intrinsics_.width = calibration.resolution_width;
  intrinsics_.height = calibration.resolution_height;

  // Scale intrinsics according to image scale
  intrinsics_.fu *= image_scale_;
  intrinsics_.fv *= image_scale_;

  // Calculate distortion map
  cv::Mat1f camera_matrix(3, 3);
  camera_matrix << param.fx, 0, param.cx, 0, param.fy, param.cy, 0, 0, 1;
  cv::Mat1f new_camera_matrix(3, 3);
  new_camera_matrix << intrinsics_.fu, 0, intrinsics_.ppu, 0, intrinsics_.fv,
      intrinsics_.ppv, 0, 0, 1;
  cv::Mat1f distortion_coeff(1, 8);
  distortion_coeff << param.k1, param.k2, param.p1, param.p2, param.k3,
      param.k4, param.k5, param.k6;
  cv::Mat map1, map2, map3;
  cv::initUndistortRectifyMap(
      camera_matrix, distortion_coeff, cv::Mat{}, new_camera_matrix,
      cv::Size{intrinsics_.width, intrinsics_.height}, CV_32FC1, map1, map2);
  cv::convertMaps(map1, map2, distortion_map_, map3, CV_16SC2, true);
}

AzureKinectDepthMulti::AzureKinectDepthMulti(  uint32_t device_index,
                                               const std::string &name,
                                               float image_scale,
                                               bool use_color_as_world_frame,
                                               float depth_offset)
    : device_index_{device_index},
      DepthCamera{name},
      image_scale_{image_scale},
      use_color_as_world_frame_{use_color_as_world_frame},
      depth_offset_{depth_offset},
      azure_kinect_{AzureKinectMulti::GetInstance()} {
  azure_kinect_.UseDepthCamera();
  azure_kinect_id_ = azure_kinect_.RegisterID(device_index_);
}

AzureKinectDepthMulti::AzureKinectDepthMulti(
    uint32_t device_index,
    const std::string &name, const std::filesystem::path &metafile_path)
    : DepthCamera{name, metafile_path},
      azure_kinect_{AzureKinectMulti::GetInstance()} {
  azure_kinect_.UseDepthCamera();
  azure_kinect_id_ = azure_kinect_.RegisterID(device_index_);
}

AzureKinectDepthMulti::~AzureKinectDepthMulti() {
  azure_kinect_.UnregisterID(device_index_, azure_kinect_id_);
}

bool AzureKinectDepthMulti::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;
  if (!initial_set_up_ && !azure_kinect_.SetUp())
    return false;
  if (use_color_as_world_frame_)
    set_camera2world_pose(*azure_kinect_.depth2color_pose(device_index_));
  GetIntrinsicsAndDistortionMap();
  SaveMetaDataIfDesired();
  set_up_ = true;
  initial_set_up_ = true;
  return UpdateImage(true);
}

void AzureKinectDepthMulti::set_image_scale(float image_scale) {
  image_scale_ = image_scale;
  set_up_ = false;
}

void AzureKinectDepthMulti::set_use_color_as_world_frame(
    bool use_color_as_world_frame) {
  use_color_as_world_frame_ = use_color_as_world_frame;
  set_up_ = false;
}

void AzureKinectDepthMulti::set_depth_offset(float depth_offset) {
  depth_offset_ = depth_offset;
}

bool AzureKinectDepthMulti::UpdateImage(bool synchronized) {
  if (!set_up_) {
    std::cerr << "Set up azure kinect depth camera " << name_ << " first"
              << std::endl;
    return false;
  }

  // Get and undistort image
  azure_kinect_.UpdateCapture(device_index_, azure_kinect_id_, synchronized);
  cv::remap(
      cv::Mat{cv::Size{intrinsics_.width, intrinsics_.height}, CV_16UC1,
                    (void *)azure_kinect_.capture(device_index_)
                        .get_depth_image()
                        .get_buffer(),
              cv::Mat::AUTO_STEP},
      image_, distortion_map_, cv::Mat{}, cv::INTER_NEAREST,
      cv::BORDER_CONSTANT);

  // Add depth offset
  if (depth_offset_) {
    short depth_value_offset = depth_offset_ / depth_scale_;
    image_ += depth_value_offset;
  }

  SaveImageIfDesired();
  return true;
}

float AzureKinectDepthMulti::image_scale() const { return image_scale_; }

bool AzureKinectDepthMulti::use_color_as_world_frame() const {
  return use_color_as_world_frame_;
}

float AzureKinectDepthMulti::depth_offset() const { return depth_offset_; }

const Transform3fA *AzureKinectDepthMulti::color2depth_pose() const {
  return azure_kinect_.color2depth_pose(device_index_);
}

const Transform3fA *AzureKinectDepthMulti::depth2color_pose() const {
  return azure_kinect_.depth2color_pose(device_index_);
}

bool AzureKinectDepthMulti::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  ReadOptionalValueFromYaml(fs, "camera2world_pose", &camera2world_pose_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_index", &save_index_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
  ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
  ReadOptionalValueFromYaml(fs, "image_scale", &image_scale_);
  ReadOptionalValueFromYaml(fs, "use_color_as_world_frame",
                            &use_color_as_world_frame_);
  ReadOptionalValueFromYaml(fs, "depth_offset", &depth_offset_);
  fs.release();

  // Process parameters
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  world2camera_pose_ = camera2world_pose_.inverse();
  return true;
}

void AzureKinectDepthMulti::GetIntrinsicsAndDistortionMap() {
  // Load intrinsics from camera
  const k4a_calibration_camera_t calibration{
      azure_kinect_.calibration(device_index_).depth_camera_calibration};
  const k4a_calibration_intrinsic_parameters_t::_param param =
      calibration.intrinsics.parameters.param;
  intrinsics_.fu = param.fx;
  intrinsics_.fv = param.fy;
  intrinsics_.ppu = param.cx;
  intrinsics_.ppv = param.cy;
  intrinsics_.width = calibration.resolution_width;
  intrinsics_.height = calibration.resolution_height;
  depth_scale_ = 0.001f;

  // Scale intrinsics according to image scale
  intrinsics_.fu *= image_scale_;
  intrinsics_.fv *= image_scale_;

  // Calculate distortion map
  cv::Mat1f camera_matrix(3, 3);
  camera_matrix << param.fx, 0, param.cx, 0, param.fy, param.cy, 0, 0, 1;
  cv::Mat1f new_camera_matrix(3, 3);
  new_camera_matrix << intrinsics_.fu, 0, intrinsics_.ppu, 0, intrinsics_.fv,
      intrinsics_.ppv, 0, 0, 1;
  cv::Mat1f distortion_coeff(1, 8);
  distortion_coeff << param.k1, param.k2, param.p1, param.p2, param.k3,
      param.k4, param.k5, param.k6;
  cv::Mat map1, map2, map3;
  cv::initUndistortRectifyMap(
      camera_matrix, distortion_coeff, cv::Mat{}, new_camera_matrix,
      cv::Size{intrinsics_.width, intrinsics_.height}, CV_32FC1, map1, map2);
  cv::convertMaps(map1, map2, distortion_map_, map3, CV_16SC2, true);
}

}  // namespace m3t
