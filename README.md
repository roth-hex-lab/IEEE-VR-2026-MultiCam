# MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects
<div align="center">
    <a href="https://arxiv.org/abs/2603.22839" target="_blank">
    <img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper arXiv"></a>
</div>

## Publication

Official code of paper GBOT: Graph-Based 3D Object Tracking for Augmented Reality-Assisted Assembly Guidance (IEEE VR 2026)

## Introduction

Multi-camera dynamic Augmented Reality (AR) applications require a camera pose estimation to leverage individual information from each camera in one common system. This can be achieved by combining contextual information, such as markers or objects, across multiple views. While commonly cameras are calibrated in an initial step or updated through the constant use of markers, another option is to leverage information already present in the scene, like known objects. Another downside of marker-based tracking is that markers have to be tracked inside the field-of-view (FoV) of the cameras.
To overcome these limitations, we propose a constant dynamic camera pose estimation leveraging spatiotemporal FoV overlaps of known objects on the fly. To achieve that, we enhance the state-of-the-art object pose estimator to update our spatiotemporal scene graph, enabling a relation even among non-overlapping FoV cameras. To evaluate our approach, we introduce a multi-camera, multi-object pose estimation dataset with temporal FoV overlap, including static and dynamic cameras. Furthermore, in FoV overlapping scenarios, we outperform the state-of-the-art on the widely used YCB-V and T-LESS dataset in camera pose accuracy. Our performance on both previous and our proposed datasets validates the effectiveness of our marker-less approach for AR applications.

<a href="https://www.youtube.com/watch?v=O-o3Y0Mzrw4">
<p align="center">
 <img src="asset/arch.pdf">
    <br> 
    <em>IEEE VR presentation</em>
</p>
</a>

## Code and dataset
Code and dataset will be published before April 10th, 2026
