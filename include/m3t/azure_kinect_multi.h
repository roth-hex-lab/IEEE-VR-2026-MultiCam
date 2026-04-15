// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef M3T_INCLUDE_M3T_AZURE_KINECT_MULTI_H_
#define M3T_INCLUDE_M3T_AZURE_KINECT_MULTI_H_

#include <filesystem/filesystem.h>
#include <m3t/camera.h>
#include <m3t/common.h>

#include <chrono>
#include <iostream>
#include <k4a/k4a.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace m3t {

/**
 * \brief Singleton class that allows getting data from a single Azure Kinect
 * instance and that is used by \ref AzureKinectColorCamera and \ref
 * AzureKinectDepthCamera.
 *
 * \details The method `UpdateCapture()` updates the `capture` object if
 * `UpdateCapture()` was already called with the same `id` before. If
 * `UpdateCapture()` was not yet called by the `id`, the same capture is used.
 * If the capture is updated, all memory values except for the `id` that called
 * the function are reset. All methods that are required to operate multiple
 * \ref Camera objects are thread-safe.
 */
class AzureKinectMulti {
 public:
  // Singleton instance getter
  static AzureKinectMulti &GetInstance();
  AzureKinectMulti(const AzureKinectMulti &) = delete;
  AzureKinectMulti &operator=(const AzureKinectMulti &) = delete;
  ~AzureKinectMulti();

  // Configuration and setup
  void UseColorCamera();
  void UseDepthCamera();
  int RegisterID(uint32_t device_index);
  bool UnregisterID(uint32_t device_index, int id);
  //bool Loaddevice(k4a::device &kdevice);
  bool SetUp();
  //k4a::device master_device{};
  //k4a::device sub_device{};
  std::vector<k4a::device> devices_{};
  //uint32_t device_index_;

  // Main methods
  bool UpdateCapture(uint32_t device_index, int id, bool synchronized);

  // Getters
  bool use_color_camera() const;
  bool use_depth_camera() const;
  const k4a::capture &capture(uint32_t device_index) const;
  const k4a::calibration &calibration(uint32_t device_index) const;
  const Transform3fA *color2depth_pose(uint32_t device_index) const;
  const Transform3fA *depth2color_pose(uint32_t device_index) const;

 private:
  AzureKinectMulti() = default;

  // Private data
  k4a_device_configuration_t config_{};
  std::vector<std::map<int, bool>> update_capture_ids_{};
  std::vector<int> next_ids_;
  int num_devices_;

  // Public data
  
  std::vector<k4a::capture> captures_{};
  std::vector<k4a::calibration> calibrations_{};
  std::vector<Transform3fA> color2depth_poses_{Transform3fA::Identity()};
  std::vector<Transform3fA> depth2color_poses_{Transform3fA::Identity()};

  // Internal state variables
  std::mutex mutex_;
  bool use_color_camera_ = false;
  bool use_depth_camera_ = false;
  bool initial_set_up_ = false;
};

/**
 * \brief \ref Camera that allows getting color images from an \ref AzureKinect
 * camera.
 *
 * @param image_scale scales images to avoid borders after rectification.
 * @param use_depth_as_world_frame specifies the depth camera frame as world
 * frame and automatically defines `camera2world_pose` as `color2depth_pose`.
 */
class AzureKinectColorMulti : public ColorCamera {
 public:
  // Constructors, destructor, and setup method
  AzureKinectColorMulti(uint32_t device_index, const std::string &name,
                        float image_scale = 1.05f,
                         bool use_depth_as_world_frame = false);
  AzureKinectColorMulti(uint32_t device_index, const std::string &name,
                         const std::filesystem::path &metafile_path);
  AzureKinectColorMulti(const AzureKinectColorMulti &) = delete;
  AzureKinectColorMulti &operator=(const AzureKinectColorMulti &) = delete;
  ~AzureKinectColorMulti();
  bool SetUp();

  // Setters
  void set_image_scale(float image_scale);
  void set_use_depth_as_world_frame(bool use_depth_as_world_frame);

  // Main method
  bool UpdateImage(bool synchronized) override;

  // Getters
  float image_scale() const;
  bool use_depth_as_world_frame() const;
  const Transform3fA *color2depth_pose() const;
  const Transform3fA *depth2color_pose() const;

 private:
  // Helper methods
  bool LoadMetaData();
  void GetIntrinsicsAndDistortionMap();

  // Data
  uint32_t device_index_;
  AzureKinectMulti &azure_kinect_;
  int azure_kinect_id_{};
  float image_scale_ = 1.05f;
  bool use_depth_as_world_frame_ = false;
  cv::Mat distortion_map_;
  bool initial_set_up_ = false;
};

/**
 * \brief \ref Camera that allows getting depth images from an \ref AzureKinect
 * camera.
 *
 * @param image_scale scales image to avoid borders after rectification.
 * @param use_color_as_world_frame specifies the color camera frame as world
 * frame and automatically define `camera2world_pose` as `depth2color_pose`.
 * @param depth_offset constant offset that is added to depth image values and
 * which is provided in meter.
 */
class AzureKinectDepthMulti : public DepthCamera {
 public:
  // Constructors, destructor, and setup method
  AzureKinectDepthMulti(uint32_t device_index, const std::string &name,
                        float image_scale = 1.0f,
                         bool use_color_as_world_frame = true,
                         float depth_offset = 0.0f);
  AzureKinectDepthMulti(uint32_t device_index, const std::string &name,
                         const std::filesystem::path &metafile_path);
  AzureKinectDepthMulti(const AzureKinectDepthMulti &) = delete;
  AzureKinectDepthMulti &operator=(const AzureKinectDepthMulti &) = delete;
  ~AzureKinectDepthMulti();
  bool SetUp() override;

  // Setters
  void set_image_scale(float image_scale);
  void set_use_color_as_world_frame(bool use_color_as_world_frame);
  void set_depth_offset(float depth_offset);

  // Main method
  bool UpdateImage(bool synchronized) override;

  // Getters
  float image_scale() const;
  bool use_color_as_world_frame() const;
  float depth_offset() const;
  const Transform3fA *color2depth_pose() const;
  const Transform3fA *depth2color_pose() const;

 private:
  // Helper methods
  bool LoadMetaData();
  void GetIntrinsicsAndDistortionMap();

  // Data
  uint32_t device_index_;
  AzureKinectMulti &azure_kinect_;
  int azure_kinect_id_{};
  float image_scale_ = 1.0f;
  bool use_color_as_world_frame_ = true;
  float depth_offset_ = 0.0f;
  cv::Mat distortion_map_;
  bool initial_set_up_ = false;
};

}  // namespace m3t

#endif  // M3T_INCLUDE_M3T_AZURE_KINECT_CAMERA_H_
