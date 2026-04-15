# MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects
<div align="center">
    <a href="https://ieeexplore.ieee.org/document/11458691" target="_blank">
    <img src="https://img.shields.io/badge/ieee-%2300629B.svg?&style=for-the-badge&logo=ieee&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2603.22839" target="_blank">
    <img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper arXiv"></a>
</div>

## Publication

Official code of paper GBOT: Graph-Based 3D Object Tracking for Augmented Reality-Assisted Assembly Guidance (IEEE VR 2026)

## Introduction

Multi-camera dynamic Augmented Reality (AR) applications require a camera pose estimation to leverage individual information from each camera in one common system. To overcome these limitations of marker-based methods, we propose a constant dynamic camera pose estimation leveraging spatiotemporal FoV overlaps of known objects on the fly. To achieve that, we enhance the state-of-the-art object pose estimator to update our spatiotemporal scene graph, enabling a relation even among non-overlapping FoV cameras. To evaluate our approach, we introduce a multi-camera, multi-object pose estimation dataset with temporal FoV overlap, including static and dynamic cameras. Furthermore, in FoV overlapping scenarios, we outperform the state-of-the-art on the widely used YCB-V and T-LESS dataset in camera pose accuracy. Our performance on both previous and our proposed datasets validates the effectiveness of our marker-less approach for AR applications.

<a href="https://www.youtube.com/watch?v=O-o3Y0Mzrw4">
<p align="center">
 <img src="asset/teaser.png">
    <br> 
    <em>IEEE VR presentation</em>
</p>
</a>

## Usage
We provide a codebase for retrieving live streams from multiple cameras, including Azure Kinect, Realsense, Hololens

- test Hololens live stream
  ```
  examples/test_hololens.cpp
  ```
- for kinect,  just change the type in
  ```
    int cam_id = 0;
    auto color_camera_ptr{
      std::make_shared<m3t::AzureKinectColorMulti>(cam_id, "azure_kinect_color")};
    auto depth_camera_ptr{std::make_shared<m3t::AzureKinectDepthMulti>(
        cam_id, "azure_kinect_depth" + std::to_string(cam_id))};
  ```
- for real sense camera,  just change the type in
  ```
    auto color_camera_ptr{
        std::make_shared<m3t::RealSenseColorCamera>("real_sense_color")};
    auto depth_camera_ptr{
        std::make_shared<m3t::RealSenseDepthCamera>("real_sense_depth")};
  ```
## Our MultiCam dataset
comming soon

## Pretrained checkpoints and config files
comming soon

## Dataset Evaluation
Evaluation multi-view setup on YCB-Video dataset:
```
project evaluate_ycb_dataset_multiview
examples/evaluate_ycb_dataset_multiview.cpp
examples/evaluate_ycb_evaluator_multiview.cpp
```

Evaluation multi-view setup on T-less dataset:
```
project evaluate_tless_dataset_multiview
examples/evaluate_tless_dataset_multiview.cpp
examples/evaluate_tless_evaluator_multiview.cpp
```

Evaluation multi-view setup on our MultiCam dataset:
```
project evaluate_scenertm6d_dataset_multiview
examples/evaluate_scenertm6d_dataset_multiview.cpp
examples/evaluate_scenertm6d_evaluator_multiview.cpp
```

## Demo
The code in `examples/demo.cpp` contains file for real-time demo.

## Acknowledgement

Our codes are partially based on previoys works. We sincerely thank [CosyPose](https://github.com/yannlabb/cosypose), [hl2ss](https://github.com/jdibenes/hl2ss), [m3t](https://github.com/DLR-RM/3DObjectTracking/tree/master/M3T) , [mmdeploy](https://github.com/open-mmlab/mmdeploy) for providing their wonderful code to the community!

## Citations
If you find MultiCam is useful in your research or applications, please consider giving us a star 🌟 and citing it.

```bibtex
@inproceedings{li2026multicam,
  title={MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects},
  author={Li, Shiyu and Schieber, Hannah and Waldow, Kristoffer and Busam, Benjamin and Kreimeier, Julian and Roth, Daniel},
  journal={IEEE transactions on visualization and computer graphics},
  year={2026},
  publisher={IEEE}
}
```

## Licence
Permission to use, copy, modify and distribute this software and its documentation for educational, research and non-profit purposes only. 
Any modification based on this work must be open-source and prohibited for commercial use.  If you need a commercial license, please feel free to contact daniel.roth@tum.de.
