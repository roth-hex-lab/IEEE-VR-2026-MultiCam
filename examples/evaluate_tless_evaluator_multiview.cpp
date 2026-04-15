// Copyright (c) HEX lab, Technical University of Munich

#include "evaluate_tless_evaluator_multiview.h"


TLESSEvaluator::TLESSEvaluator(
    int num_view, const std::string &name,
    const std::filesystem::path &model_directory,
    const std::filesystem::path &dataset_directory,
    const std::filesystem::path &external_directory,
    const std::vector<int> &sequence_ids,
    const std::vector<std::string> &evaluated_body_names)
    : num_view_{num_view},
      name_{name}, model_directory_{model_directory},
      dataset_directory_{dataset_directory},
      external_directory_{external_directory},
      sequence_ids_{sequence_ids},
      evaluated_body_names_{evaluated_body_names}
     {
  // Compute thresholds used to compute AUC score
  float threshold_step = kThresholdMax / float(kNCurveValues);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    thresholds_[i] = threshold_step * (0.5f + float(i));
  }
  
}

bool TLESSEvaluator::SetUp() {
  set_up_ = false;

  if (!CreateRunConfigurations()) return false;
  AssembleTrackedBodyNames();
  std::cout << "load bodies" << std::endl;
  if (!LoadBodies()) return false;
  std::cout << "generate models" << std::endl;
  if (!GenerateModels()) return false;
  std::cout << "load key frames" << std::endl;
  if (!LoadKeyframes()) return false;
  std::cout << "load key frames done" << std::endl;
  GenderateReducedVertices();
  GenerateKDTrees();

  set_up_ = true;
  return true;
}

void TLESSEvaluator::set_detector_folder(const std::string &detector_folder) {
  detector_folder_ = detector_folder;
}

void TLESSEvaluator::set_evaluate_refinement(bool evaluate_refinement) {
  evaluate_refinement_ = evaluate_refinement;
}

void TLESSEvaluator::set_use_detector_initialization(
    bool use_detector_initialization) {
  use_detector_initialization_ = use_detector_initialization;
}

void TLESSEvaluator::set_use_matlab_gt_poses(bool use_matlab_gt_poses) {
  use_matlab_gt_poses_ = use_matlab_gt_poses;
}

void TLESSEvaluator::set_run_sequentially(bool run_sequentially) {
  run_sequentially_ = run_sequentially;
}

void TLESSEvaluator::set_use_random_seed(bool use_random_seed) {
  use_random_seed_ = use_random_seed;
}

void TLESSEvaluator::set_n_vertices_evaluation(int n_vertices_evaluation) {
  n_vertices_evaluation_ = n_vertices_evaluation;
}

void TLESSEvaluator::set_visualize_tracking(bool visualize_tracking) {
  visualize_tracking_ = visualize_tracking;
}

void TLESSEvaluator::set_visualize_frame_results(bool visualize_frame_results) {
  visualize_frame_results_ = visualize_frame_results;
}

void TLESSEvaluator::StartSavingImages(
    const std::filesystem::path &save_directory) {
  save_images_ = true;
  save_directory_ = save_directory;
}

void TLESSEvaluator::StopSavingImages() { save_images_ = false; }

void TLESSEvaluator::set_use_multi_region(bool use_multi_region) {
  use_multi_region_ = use_multi_region;
}

void TLESSEvaluator::set_use_region_modality(bool use_region_modality) {
  use_region_modality_ = use_region_modality;
}

void TLESSEvaluator::set_use_depth_modality(bool use_depth_modality) {
  use_depth_modality_ = use_depth_modality;
}

void TLESSEvaluator::set_use_texture_modality(bool use_texture_modality) {
  use_texture_modality_ = use_texture_modality;
}

void TLESSEvaluator::set_measure_occlusions_region(
    bool measure_occlusions_region) {
  measure_occlusions_region_ = measure_occlusions_region;
}

void TLESSEvaluator::set_measure_occlusions_depth(bool measure_occlusions_depth) {
  measure_occlusions_depth_ = measure_occlusions_depth;
}

void TLESSEvaluator::set_measure_occlusions_texture(
    bool measure_occlusions_texture) {
  measure_occlusions_texture_ = measure_occlusions_texture;
}

void TLESSEvaluator::set_model_occlusions_region(bool model_occlusions_region) {
  model_occlusions_region_ = model_occlusions_region;
}

void TLESSEvaluator::set_model_occlusions_depth(bool model_occlusions_depth) {
  model_occlusions_depth_ = model_occlusions_depth;
}

void TLESSEvaluator::set_model_occlusions_texture(bool model_occlusions_texture) {
  model_occlusions_texture_ = model_occlusions_texture;
}

void TLESSEvaluator::set_tracker_setter(
    const std::function<void(std::shared_ptr<m3t::Tracker>)> &tracker_setter) {
  tracker_setter_ = tracker_setter;
}

void TLESSEvaluator::set_refiner_setter(
    const std::function<void(std::shared_ptr<m3t::Refiner>)> &refiner_setter) {
  refiner_setter_ = refiner_setter;
}

void TLESSEvaluator::set_optimizer_setter(
    const std::function<void(std::shared_ptr<m3t::Optimizer>)>
        &optimizer_setter) {
  optimizer_setter_ = optimizer_setter;
}

void TLESSEvaluator::set_region_modality_setter(
    const std::function<void(std::shared_ptr<m3t::RegionModality>)>
        &region_modality_setter) {
  region_modality_setter_ = region_modality_setter;
}

void TLESSEvaluator::set_region_model_setter(
    const std::function<void(std::shared_ptr<m3t::RegionModel>)>
        &region_model_setter) {
  region_model_setter_ = region_model_setter;
  set_up_ = false;
}

void TLESSEvaluator::set_depth_modality_setter(
    const std::function<void(std::shared_ptr<m3t::DepthModality>)>
        &depth_modality_setter) {
  depth_modality_setter_ = depth_modality_setter;
}

void TLESSEvaluator::set_depth_model_setter(
    const std::function<void(std::shared_ptr<m3t::DepthModel>)>
        &depth_model_setter) {
  depth_model_setter_ = depth_model_setter;
  set_up_ = false;
}

void TLESSEvaluator::set_texture_modality_setter(
    const std::function<void(std::shared_ptr<m3t::TextureModality>)>
        &texture_modality_setter) {
  texture_modality_setter_ = texture_modality_setter;
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
    //std::cout << vec_n.size() << std::endl;
    if (vec_n.size() > n_max) n_max = vec_n.size();
  }
  //std::cout << n_max << std::endl;
  // creat an empty tensor in the shape of (bsz, n_max, 4, 4)
  torch::Tensor tensor_padded = torch::zeros({bsz, n_max, 4, 4});
 
  // tensor_list_padded = [];
  for (int vec_id = 0; vec_id < vec_list.size(); vec_id++) {
    auto vec_n = vec_list[vec_id];
    //for (auto vec : vec_n) std::cout << vec.matrix() << std::endl;
    int n_pad = n_max - vec_n.size();
    //std::cout << n_pad << std::endl;
    if (n_pad > 0) {
      std::vector<m3t::Transform3fA> pad;
      
      //std::sample(vec_n.begin(), vec_n.end(), std::back_inserter(pad), n_pad, std::mt19937{std::random_device{}()});
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
      //for (auto index : indices) std::cout << index << std::endl;
      std::transform(cbegin(indices), cend(indices), std::back_inserter(pad),
                     [&vec_n](auto const index) {
                       return *std::next(cbegin(vec_n), index);
                       // note, for std::vector or std::array of samples, you
                       // can use return samples[index];
                     });
      //for (auto vec : pad) std::cout << vec.matrix() << std::endl;
      //std::cout << pad.size() << std::endl;
      // ids_pad = np_random.choice(np.arange(len(vec_n)), size = n_pad);
      // pad = vec_n[ids_pad];

      // tensor_n_padded = torch.cat((vec_n, pad), dim = 0);
      vec_n.insert(vec_n.end(), pad.begin(), pad.end());
    }
    //std::cout << vec_n.size() << std::endl;
    for (int id = 0; id < vec_n.size(); id++) {
      auto vec = vec_n[id].matrix();
      //std::cout << vec << std::endl;
      torch::Tensor T;
      Eigen::MatrixXf mat = vec;
      EigenMat2Tensor(T, mat);
      //std::vector<int64_t> dims = {vec.rows(), vec.cols()};
      //torch::Tensor T = torch::from_blob(vec.data(), dims).clone();
      //std::cout << T << std::endl;
      tensor_padded.index({vec_id, id}) = T;
      // tensor_list_padded.append(tensor_n_padded);
    }
  }
  return tensor_padded;
}

bool TLESSEvaluator::Evaluate() {
  if (!set_up_) {
    std::cerr << "Set up evaluator " << name_ << " first" << std::endl;
    return false;
  }
  if (run_configurations_.empty()) return false;
  
  for (int obj_id = 0; obj_id < evaluated_body_names_.size(); obj_id++)
    body_name2idx_map_.insert({evaluated_body_names_[obj_id], obj_id+1});

  // read model infos
  std::cout << "read model infos" << std::endl;
  std::filesystem::path model_path{dataset_directory_ / "models" /
                                   "models_info.json"};
  cv::FileStorage fs_model;

  try {
    fs_model.open(cv::samples::findFile(model_path.string()),
                  cv::FileStorage::READ);
  } catch (cv::Exception e) {
  }
  if (!fs_model.isOpened()) {
    std::cerr << "Could not open file " << model_path << std::endl;
    return false;
  }
  
  symmetry_ids.clear();
  symmetries_.clear();
  symmetries_.resize(body_name2idx_map_.size());
  for (const auto &[body_name, idx] : body_name2idx_map_) {
    //std::cout << body_name << std::endl;
    // read object model infos
    cv::FileNode fn_object{fs_model[std::to_string(idx)]};
    cv::FileNode fn_discrete{fn_object["symmetries_discrete"]};
    cv::FileNode fn_continuous{fn_object["symmetries_continuous"]};
    if (!fn_continuous.empty() || !fn_discrete.empty())
      symmetry_ids.push_back(idx-1);

    
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
        //std::cout << sym_matrix.matrix() << std::endl;
        //symmetries_[idx-1].push_back(sym_matrix);
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
          //std::cout << sym_matrix.matrix() << std::endl;
          //symmetries_[idx - 1].push_back(sym_matrix);
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
                  
      }
      else {
          all_M.push_back(sym_d);
        }       
    }
    symmetries_[idx - 1] = all_M;
    // read key points data
    std::ifstream accfile;
    std::filesystem::path path_kpts{dataset_directory_ /"models_kpts"/ (body_name + ".txt")};
    // std::cout << path_kpts << std::endl;
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
      }
    }
    if (accfile.is_open()) {
      accfile.close();
    }

    kpts_gt_.push_back(numbers);
  }

  //prepare mesh database
  int num_obj = kpts_gt_.size();
  int num_kpts = kpts_gt_[0].size();
  kpts_3d = torch::zeros({num_obj, num_kpts, 3});
  for (int obj_id = 0; obj_id < num_obj; obj_id++) {
    for (int kpts_id = 0; kpts_id < num_kpts; kpts_id++) {
      kpts_3d.index({obj_id, kpts_id, 0}) = kpts_gt_[obj_id][kpts_id].x;
      kpts_3d.index({obj_id, kpts_id, 1}) = kpts_gt_[obj_id][kpts_id].y;
      kpts_3d.index({obj_id, kpts_id, 2}) = kpts_gt_[obj_id][kpts_id].z;
    }
  }
  //std::cout << kpts_3d << std::endl;
  //std::cout << "obj 0" << std::endl;
  //std::cout << kpts_3d.index({0}) << std::endl;
  //std::cout << "generate sym tensor" << std::endl;
  sym_tensor = pad_stack_tensors(symmetries_);
  //std::cout << sym_tensor << std::endl;
  //std::cout << "generate sym tensor done" << std::endl;
  for (int obj_id = 0; obj_id < num_obj; obj_id++) {
    n_sym_mapping[obj_id] = symmetries_[obj_id].size();
  }
  mesh_db.kpts_3d = kpts_3d;
  mesh_db.n_sym_mapping = n_sym_mapping;
  mesh_db.sym_tensor = sym_tensor;
  //std::cout << symmetries_[1].size()<< std::endl; 

  //for (int iter = 0; iter < symmetry_ids.size(); iter++)
  //  std::cout << symmetry_ids[iter] << std::endl;

  // Initialize pose detector
  int status{};
  auto device_name = "cuda";
  status = mmdeploy_pose_detector_create_by_path(
      model_directory_.generic_string().c_str(), device_name, 0,
      &pose_detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create pose_estimator, code: %d\n", (int)status);
    return 1;
  }
  std::cout << "Pose detector loaded" << std::endl;
  // Evaluate all run configurations
  results_.clear();
  final_results_.clear();
  if (run_sequentially_ || visualize_tracking_ || visualize_frame_results_) {
    auto renderer_geometry_ptr{std::make_shared<m3t::RendererGeometry>("rg")};
    renderer_geometry_ptr->SetUp();
    for (size_t i = 0; i < int(run_configurations_.size()); ++i) {
      std::vector<SequenceResult> sequence_results;
      
      if (!EvaluateRunConfiguration(run_configurations_[i],
                                    renderer_geometry_ptr, &sequence_results))
        continue;
      results_.insert(end(results_), begin(sequence_results),
                      end(sequence_results));
      for (const auto &sequence_result : sequence_results) {
        std::string title{sequence_result.sequence_name + ": "};
        VisualizeResult(sequence_result.average_result, title);
      }
    }
  } else {
    std::vector<std::shared_ptr<m3t::RendererGeometry>> renderer_geometry_ptrs(
        omp_get_max_threads());
    for (auto &renderer_geometry_ptr : renderer_geometry_ptrs) {
      renderer_geometry_ptr = std::make_shared<m3t::RendererGeometry>("rg");
      renderer_geometry_ptr->SetUp();
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(run_configurations_.size()); ++i) {
      std::vector<SequenceResult> sequence_results;
      if (!EvaluateRunConfiguration(
              run_configurations_[i],
              renderer_geometry_ptrs[omp_get_thread_num()], &sequence_results))
        continue;
#pragma omp critical
      {
        results_.insert(end(results_), begin(sequence_results),
                        end(sequence_results));
        for (const auto &sequence_result : sequence_results) {
          std::string title{sequence_result.sequence_name + ": " +
                            sequence_result.body_name};
          VisualizeResult(sequence_result.average_result, title);
        }
      }
    }
  }

  // Calculate sequence results
  std::cout << std::endl << std::string(80, '-') << std::endl;
  for (int sequence_id : sequence_ids_) {
    auto sequence_name{SequenceIDToName(sequence_id)};
    Result result{CalculateAverageSequenceResult({sequence_name})};
    VisualizeResult(result, sequence_name);
  }

  // Calculate body results
  /*
  std::cout << std::endl << std::string(80, '-') << std::endl;
  for (const auto &body_name : evaluated_body_names_) {
    Result result{CalculateAverageBodyResult({body_name})};
    VisualizeResult(result, body_name);
    final_results_.insert({body_name, std::move(result)});
  }
  */

  // Calculate average results
  Result final_result_all{CalculateAverageBodyResult(evaluated_body_names_)};
  VisualizeResult(final_result_all, "all");
  final_results_.insert({"all", std::move(final_result_all)});
  std::cout << std::string(80, '-') << std::endl;
  return true;
}

void TLESSEvaluator::SaveResults(std::filesystem::path path) const {
  std::filesystem::path runtime_path = {path / ("runtime_view_" + 
                                        std::to_string(num_view_) +".txt")};
  std::ofstream ofs{runtime_path};
  for (auto const &[body_name, result] : final_results_) {
    ofs << body_name << "," << result.add_auc << "," << result.adds_auc << ","
        << result.execution_times.complete_cycle << ","
        << result.execution_times.pose_estimation << ","
        << result.execution_times.candidate_matching << ","
        << result.execution_times.bundle_adjustment << std::endl;
  }
  ofs.flush();
  ofs.close();
}

float TLESSEvaluator::add_auc() const { return final_results_.at("all").add_auc; }

float TLESSEvaluator::adds_auc() const {
  return final_results_.at("all").adds_auc;
}

float TLESSEvaluator::execution_time() const {
  return final_results_.at("all").execution_times.complete_cycle;
}

std::map<std::string, TLESSEvaluator::Result> TLESSEvaluator::final_results()
    const {
  return final_results_;
}

Eigen::MatrixXf toEigenMatrix(std::vector<std::vector<float>> matrix) {
  Eigen::MatrixXf outMat(matrix.size(), matrix[0].size());
  for (int i = 0; i < matrix.size(); i++) {
    outMat.row(i) = Eigen::VectorXf::Map(&matrix[i][0], matrix[0].size());
  }
  return outMat;
}


void global_scene_fusion(
    cosypose::Ransac_candidates &candidates,
                         std::map<int, m3t::Transform3fA> &body2world_poses_map,
    std::map<int, m3t::Transform3fA> TWC, std::vector<int> inliers_id,
    std::vector<int> &inlier_labels) {
  std::vector<int> view_ids = candidates.view_ids;
  std::vector<int> obj_ids = candidates.labels;
  torch::Tensor object_pose_tensors = candidates.poses;
  m3t::Transform3fA obj_pose, cam_pose;
  for (int detect_id = view_ids.size()-1; detect_id > -1; detect_id--) {

    if (TWC.find(view_ids[detect_id]) != TWC.end() 
        && std::count(inliers_id.begin(), inliers_id.end(), detect_id) > 0
        ) {
      cam_pose = TWC[view_ids[detect_id]];
      Eigen::MatrixXf E = Eigen::Matrix4f::Zero();
      Tensor2EigenMat(object_pose_tensors.index({detect_id}), E);
      obj_pose.matrix() = E;
      body2world_poses_map[obj_ids[detect_id]] = cam_pose*obj_pose;
      inlier_labels.push_back(obj_ids[detect_id]);
    }

  }
}

static std::string toString(const Eigen::MatrixXf &mat) {
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
                               " ", " ", "", "", "", "");
  std::stringstream ss;
  ss << mat.format(CommaInitFmt);
  std::string my_str = ss.str();
  // std::remove(my_str.begin(), my_str.end(), ' ');
  return my_str;
}

void TLESSEvaluator::SaveCamPoses(const Result &result, const std::string &title) {
  std::ofstream ofs;
  std::filesystem::path result_campose_path_{
      dataset_directory_ / "result" /
      ("scenertm6dcamview" + std::to_string(num_view_) + "before_tless-test.csv")};

  ofs.open(result_campose_path_, std::fstream::app);
  for (int index = 0; index < result.add_camposes_before.size(); index++) {
    ofs << title << "," << result.add_camposes_before[index] << std::endl;
  }
  ofs.flush();
  ofs.close();
  // std::cout << " Save camera poses done!" << std::endl;
}

void TLESSEvaluator::SavePoses(const Result &result, const std::string &title) {
  std::filesystem::path result_objpose_path_ {
    dataset_directory_ / "result" / ("scenertm6dobjview" +
        std::to_string(num_view_) +"_tless-test.csv")};
  std::ofstream ofs;
  ofs.open(result_objpose_path_, std::fstream::app);
  for (int index = 0; index < result.add_objposes.size(); index++) {
    ofs << title << "," << std::setfill('0') << std::setw(6)
        << result.frame_index << "," << result.add_objposes[index] << ","
        << result.execution_times.complete_cycle / 1000 << std::endl;
  }
  ofs.flush();
  ofs.close();
  //std::cout << " Save obj poses done!" << std::endl;

  std::filesystem::path result_campose_path_{dataset_directory_ / "result" /
                                             ("scenertm6dcamview" +
        std::to_string(num_view_) + "after_tless-test.csv")};

  ofs.open(result_campose_path_, std::fstream::app);
  for (int index = 0; index < result.add_camposes.size(); index++) {
    ofs << title << "," << result.add_camposes[index] << std::endl;
  }
  ofs.flush();
  ofs.close();
  //std::cout << " Save camera poses done!" << std::endl;
}

bool TLESSEvaluator::EvaluateRunConfiguration(
    const RunConfiguration &run_configuration,
    const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
    std::vector<SequenceResult> *sequence_results) {
  const auto &sequence_name{run_configuration.sequence_name};
  const auto &evaluated_body_names{run_configuration.evaluated_body_names};
  const auto &tracked_body_names{run_configuration.tracked_body_names};
  //std::cout << sequence2keyframes_map_.size() << std::endl;
  //std::cout << sequence_name << std::endl;
  const auto &keyframes{sequence2keyframes_map_.at(sequence_name)};
  
  size_t n_keyframes = keyframes.size();
  std::chrono::high_resolution_clock::time_point begin_time;
  
  // Initialize tracker
  std::shared_ptr<m3t::Tracker> tracker_ptr;
  if (!SetUpTracker(run_configuration, renderer_geometry_ptr, &tracker_ptr))
    return false;
  std::cout << "Tracker loaded" << std::endl;
  // Read gt poses and detector poses
  //std::vector<std::vector<m3t::Transform3fA>> body2world_poses_all;
  cosypose::Ransac_candidates candidates;
  std::vector<m3t::Transform3fA> body2world_poses;
  std::map<std::string, std::vector<bool>> body_detected;
  std::cout << "Load camera and object poses" << std::endl;
  std::cout << "Evaluating scene: " + sequence_name << std::endl;
  if (!LoadGTPoses(sequence_name, &gt_body2world_poses_sequence,
                   &gt_cam2world_poses_sequence))
    return false;
  //std::cout << "camera and object poses loaded" << std::endl;
  
  // Init results
  sequence_results->resize(1);
  (*sequence_results)[0].sequence_name = sequence_name;
  (*sequence_results)[0].body_name = evaluated_body_names[0];
  (*sequence_results)[0].frame_results.clear();
  /*
  sequence_results->resize(evaluated_body_names.size());
  for (int i = 0; i < evaluated_body_names.size(); ++i) {
    (*sequence_results)[i].sequence_name = sequence_name;
    (*sequence_results)[i].body_name = evaluated_body_names[i];
    (*sequence_results)[i].frame_results.clear();
  }
  */
  Eigen::Matrix4f Trans;
  std::vector<cv::Mat> images;
  std::vector<cv::Matx33d> cam_Ks;
  std::vector<m3t::Transform3fA> cam2worlds(num_view_);
  // warmup
  std::vector<mmdeploy_mat_t> mats;
  cv::Mat img = cv::Mat::ones(720, 540, CV_8UC3);
  mmdeploy_mat_t mat{img.data,
                     img.rows,
                     img.cols,
                     3,
                     MMDEPLOY_PIXEL_FORMAT_BGR,
                     MMDEPLOY_DATA_TYPE_UINT8};
  mats.push_back(mat);
  int status{};
  for (int i = 0; i < 20; ++i) {
    mmdeploy_pose_detection_t *res{};
    status = mmdeploy_pose_detector_apply(pose_detector, mats.data(), 1, &res);
    mmdeploy_pose_detector_release_result(res, mats.size());
  }
  
  // Iterate over all frames
  for (int i = 0; i < n_keyframes; ++i) {
    Result general_results;
    
    images.clear();
    cam_Ks.clear();
    general_results.execution_times.pose_estimation = 0.0f;
    general_results.execution_times.candidate_matching = 0.0f;
    general_results.execution_times.bundle_adjustment = 0.0f;
    //std::cout << keyframes[i].size() << std::endl;
    //std::cout << keyframes[i][0] << std::endl;
    UpdateCameras(keyframes[i]);
    general_results.frame_index = keyframes[i][0];
    // Execute different evaluations
    for (auto camera_ptr : color_camera_ptrs) {
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
    begin_time = std::chrono::high_resolution_clock::now();
    Posedetector(pose_detector, images, cam_Ks, candidates,
                 body_name2idx_map_.size());
    general_results.execution_times.pose_estimation = ElapsedTime(begin_time);
    
    begin_time = std::chrono::high_resolution_clock::now();
    
    general_results.execution_times.bundle_adjustment = ElapsedTime(begin_time);

    begin_time = std::chrono::high_resolution_clock::now();
    //std::vector<float> error_views;
    //error_views.clear();
    // Initialize camera poses and object poses
    /*
    for (int cam_id = 1; cam_id < color_camera_ptrs.size(); cam_id++) {
      // Camera calibration

      //float error;
      
      //calculate_cam_poses(body2world_poses_all[0], body2world_poses_all[cam_id], Trans, error);
      //error_views.push_back(error);
      
      cam2worlds[cam_id].matrix() = Trans;
      color_camera_ptrs[cam_id]->set_camera2world_pose(cam2worlds[cam_id]); 

      depth_camera_ptrs[cam_id]->set_camera2world_pose(cam2worlds[cam_id]);
      

      for (int obj_id = 0; obj_id < body_name2idx_map_.size(); obj_id++) {
        body2world_poses_all[cam_id][obj_id].matrix() =
            cam2worlds[cam_id].matrix() *
            body2world_poses_all[cam_id ][obj_id].matrix();
      }
  }
  */
    int model_bsz = 1e3;
    int score_bsz = 1e5;
    float dist_threshold = 0.01;
    int n_ransac_iter = 2000;
    int n_min_inliers = 3;
    std::vector<int> inlier_ids;
    cosypose::Results cam_poses_results;
    //std::cout << "candidates matching begin" << std::endl;

    cosypose::multiview_candidate_matching(
        candidates, mesh_db, cam_poses_results, inlier_ids, model_bsz,
        score_bsz, dist_threshold, n_ransac_iter, n_min_inliers);

    //std::cout << "candidates matching done" << std::endl;
    using ViewPair = std::tuple<int, int>;

    std::map<ViewPair, m3t::Transform3fA> T_camera_map;
    for (int cam_idx = 0; cam_idx < cam_poses_results.view1.size(); cam_idx++) {
      int v1 = cam_poses_results.view1[cam_idx];
      int v2 = cam_poses_results.view2[cam_idx];
      //std::cout << v1 << v2 << std::endl;
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

    //Initialize views
    int n_pass = 20;
    int n = 0;
    std::map<int, m3t::Transform3fA> TWC;
    m3t::Transform3fA camera2world_pose{m3t::Transform3fA::Identity()};
    TWC[0] = camera2world_pose; // initialize main camera with identity matrix
    std::vector<int> views_initialized = {0};
    std::vector<int> views_to_initialize(num_view_-1), views_ordered(num_view_);
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

            if (std::count(keys.begin(), keys.end(), view_pair)>0) {
              m3t::Transform3fA TC2C1 = T_camera_map[view_pair];
              m3t::Transform3fA TWC2 = TWC[v2];
              TWC[v1] = TWC2 * TC2C1;
              Eigen::AngleAxisf aa(TWC[v1].rotation());
              float angle = aa.angle();
              if (angle != 0) {
                views_to_initialize.erase(std::find(views_to_initialize.begin(),
                                                    views_to_initialize.end(),
                                                    v1));
                color_camera_ptrs[v1]->set_camera2world_pose(TWC[v1]);

                depth_camera_ptrs[v1]->set_camera2world_pose(TWC[v1]);
                views_initialized.push_back(v1);
                break;
              } else
                continue;
              
            }
          }
        }

      }
      n += 1;
      if (n >= n_pass) {
        //std::cerr << "Cannot find an initialization" <<std::endl;
        break;
      }
    }
    // Fuse object poses from different views
    std::map<int, m3t::Transform3fA> body2world_poses_map;
    std::vector<int> inlier_labels;
    global_scene_fusion(candidates, body2world_poses_map, TWC, inlier_ids,
                        inlier_labels);
    general_results.execution_times.candidate_matching = ElapsedTime(begin_time);
    
    ResetBodies(tracker_ptr, body2world_poses_map);
    //std::cout << views_initialized << std::endl;
    for (int cam_id = 1; cam_id < num_view_; ++cam_id) {
      if (std::count(views_initialized.begin(), views_initialized.end(),
                     cam_id) > 0) {
      
      // std::cout << gt_cam2world_poses_sequence[0].size() << std::endl;
      m3t::Transform3fA cam_world =
          gt_cam2world_poses_sequence[i][0] *
          color_camera_ptrs[cam_id]->camera2world_pose();
      std::string frame_cam =
          std::to_string(color_camera_ptrs[cam_id]->load_index() - 1);
      int n_zeros = std::max(6 - int(frame_cam.length()), 0);
      general_results.add_camposes_before.push_back(
          std::string(n_zeros, '0') + frame_cam + "," +
          toString(cam_world.rotation()) + "," +
          toString(1000 * cam_world.translation()));}
    }
    SaveCamPoses(general_results, sequence_name);

    begin_time = std::chrono::high_resolution_clock::now();
    //std::cout << "BA  " << std::endl;
    
    ExecuteMeasuredTrackingCycle(tracker_ptr, color_camera_ptrs,
                                 depth_camera_ptrs, link_ptrs, cam_link_ptrs,
                                 views_initialized, inlier_labels, i);
    //std::cout << "BA done "<< std::endl;
    general_results.execution_times.bundle_adjustment +=
        ElapsedTime(begin_time);
    general_results.execution_times.complete_cycle =
        general_results.execution_times.pose_estimation +
        general_results.execution_times.candidate_matching +
        general_results.execution_times.bundle_adjustment;
    
    // save poses
    if (views_initialized.size() > 1) {
      for (const auto &[body_name, idx] : body_name2idx_map_) {
        const auto &body_ptr = *std::find_if(
            begin(tracker_ptr->body_ptrs()), end(tracker_ptr->body_ptrs()),
            [&](const auto &b) { return b->name() == body_name; });

        if (body_ptr->body2world_pose().translation()[2] > 0)
          general_results.add_objposes.push_back(
              std::to_string(std::stoi(body_ptr->name().substr(4, 6))) +
              ",-1," + toString(body_ptr->body2world_pose().rotation()) + "," +
              toString(1000 * body_ptr->body2world_pose().translation()));
      }
    }

    
    
    for (int cam_id = 1; cam_id < num_view_; ++cam_id) {
      if (std::count(views_initialized.begin(), views_initialized.end(),
                     cam_id) > 0) {
        m3t::Transform3fA cam_world =
            gt_cam2world_poses_sequence[i][0] *
            color_camera_ptrs[cam_id]->camera2world_pose();
        std::string frame_cam =
            std::to_string(color_camera_ptrs[cam_id]->load_index() - 1);
        int n_zeros = std::max(6 - int(frame_cam.length()), 0);
        general_results.add_camposes.push_back(
            std::string(n_zeros, '0') + frame_cam + "," +
            toString(cam_world.rotation()) + "," +
            toString(1000 * cam_world.translation()));
      }
    }
    SavePoses(general_results, sequence_name);
    
    // Calculate results
    /*
    for (int j = 0; j < evaluated_body_names.size(); ++j) {
      const auto &body_name{evaluated_body_names[j]};
      Result result{general_results};
      CalculatePoseResults(tracker_ptr, body_name,
                           gt_body2world_poses_sequence[i][j], &result);
      if (visualize_frame_results_)
        VisualizeResult(result, sequence_name + ": " + body_name);
      (*sequence_results)[j].frame_results.push_back(std::move(result));
    }
    */
    (*sequence_results)[0].frame_results.push_back(std::move(general_results));

    for (auto viewer_ptr : tracker_ptr->viewer_ptrs()) viewer_ptr->SetUp();
    if (!tracker_ptr->UpdateViewers(i)) return false;
  }
  
  // Calculate Average Results
  for (auto &sequence_result : *sequence_results)
    sequence_result.average_result =
        CalculateAverageResult(sequence_result.frame_results);
  return true;
}

bool TLESSEvaluator::SetUpTracker(
    const RunConfiguration &run_configuration,
    const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
    std::shared_ptr<m3t::Tracker> *tracker_ptr) {
  
  renderer_geometry_ptr->ClearBodies();
  *tracker_ptr = std::make_shared<m3t::Tracker>("tracker");
  cam_link_ptrs.clear();
  color_camera_ptrs.clear();
  depth_camera_ptrs.clear();
  color_depth_renderer_ptrs.clear();
  depth_depth_renderer_ptrs.clear();
  color_silhouette_renderer_ptrs.clear();
  body_ptrs.clear();
  link_ptrs.clear();
  refiner_ptrs.clear();
  optimizer_ptrs.clear();
  // Init cameras
  std::filesystem::path color_camera_directory{
      dataset_directory_ / "test_primesense" / run_configuration.sequence_name / "rgb"};
  std::filesystem::path depth_camera_directory{
      dataset_directory_ / "test_primesense" / run_configuration.sequence_name / "depth"};
  cam_link_ptrs.resize(num_view_);
  optimizer_ptrs.resize(num_view_);
  for (int cam_id = 0; cam_id < num_view_; ++cam_id) {
    cam_link_ptrs[cam_id].resize(body_name2idx_map_.size());
    // Define pose refiner for each camera view and optimize object poses separately
    auto refiner_ptr{
        std::make_shared<m3t::Refiner>("refiner" + std::to_string(cam_id))};
    refiner_ptrs.push_back(refiner_ptr);
    auto color_camera_ptr{std::make_shared<m3t::LoaderColorCamera>(
        "color_camera" + std::to_string(cam_id), color_camera_directory,
        kTLESSIntrinsics, "", 1, 6, "")};
    color_camera_ptr->SetUp();
    auto depth_camera_ptr{std::make_shared<m3t::LoaderDepthCamera>(
        "depth_camera" + std::to_string(cam_id), depth_camera_directory,
        kTLESSIntrinsics, 0.0001f, "", 1, 6, "")};
    depth_camera_ptr->SetUp();
    color_camera_ptrs.push_back(color_camera_ptr);
    depth_camera_ptrs.push_back(depth_camera_ptr);

    // Init visualizer
    auto save_directory_sequence{save_directory_ /
                                 run_configuration.sequence_name};
    if (save_images_)
      std::filesystem::create_directories(save_directory_sequence);
    if ((use_region_modality_ || use_texture_modality_) &&
        (visualize_tracking_ || save_images_)) {
      auto color_viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
          "color_viewer" + std::to_string(cam_id), color_camera_ptr,
          renderer_geometry_ptr)};
      color_viewer_ptr->set_display_images(visualize_tracking_);
      if (save_images_)
        color_viewer_ptr->StartSavingImages(save_directory_sequence);
      color_viewer_ptr->SetUp();
      (*tracker_ptr)->AddViewer(color_viewer_ptr);
    }

    if (use_depth_modality_ && (visualize_tracking_ || save_images_)) {
      auto depth_viewer_ptr{std::make_shared<m3t::NormalDepthViewer>(
          "depth_viewer" + std::to_string(cam_id), depth_camera_ptr,
          renderer_geometry_ptr, 0.3f, 1.0f)};
      depth_viewer_ptr->set_display_images(visualize_tracking_);
      if (save_images_)
        depth_viewer_ptr->StartSavingImages(save_directory_sequence);
      depth_viewer_ptr->SetUp();
      (*tracker_ptr)->AddViewer(depth_viewer_ptr);
    }

    if (model_occlusions_region_) {
      auto color_depth_renderer_ptr{
          std::make_shared<m3t::FocusedBasicDepthRenderer>(
              "color_depth_renderer" + std::to_string(cam_id),
              renderer_geometry_ptr, color_camera_ptr)};
      color_depth_renderer_ptrs.push_back(color_depth_renderer_ptr);
    }
    if (model_occlusions_depth_) {
      auto depth_depth_renderer_ptr{
          std::make_shared<m3t::FocusedBasicDepthRenderer>(
              "depth_depth_renderer" + std::to_string(cam_id),
              renderer_geometry_ptr, depth_camera_ptr)};
      depth_depth_renderer_ptrs.push_back(depth_depth_renderer_ptr);
    }
    if (use_texture_modality_) {
      auto color_silhouette_renderer_ptr{
          std::make_shared<m3t::FocusedSilhouetteRenderer>(
              "silhouette_renderer" + std::to_string(cam_id),
              renderer_geometry_ptr, color_camera_ptr, m3t::IDType::BODY)};
      color_silhouette_renderer_ptrs.push_back(color_silhouette_renderer_ptr);
    }
  }
  
  // Iterate over tracked bodies
  int obj_id = 0;
  for (const auto &body_name : run_configuration.tracked_body_names) {
    // Init body
    auto body_ptr{
        std::make_shared<m3t::Body>(*body2body_ptr_map_.at(body_name))};
    body_ptrs.push_back(body_ptr);
    renderer_geometry_ptr->AddBody(body_ptr);
    
    // Init link
    auto link_ptr{std::make_shared<m3t::Link>(
        body_name + "_link", body_ptr)};

    for (auto color_depth_renderer_ptr : color_depth_renderer_ptrs) {
      color_depth_renderer_ptr->AddReferencedBody(body_ptr);
      color_depth_renderer_ptr->SetUp();
    }

    for (auto depth_depth_renderer_ptr : depth_depth_renderer_ptrs) {
      depth_depth_renderer_ptr->AddReferencedBody(body_ptr);
      depth_depth_renderer_ptr->SetUp();
    }

    for (auto color_silhouette_renderer_ptr : color_silhouette_renderer_ptrs) {
      color_silhouette_renderer_ptr->AddReferencedBody(body_ptr);
      color_silhouette_renderer_ptr->SetUp();
    }
      
    for (int cam_id = 0; cam_id < num_view_; ++cam_id) {
      
      //link_ptrs.push_back(link_ptr);
      // Init region modality
      if (use_region_modality_) {
        float max_contour_length = 0.0f;
        for (auto &region_model_ptr : body2region_model_ptrs_map_.at(body_name))
          max_contour_length = std::max(max_contour_length,
                                        region_model_ptr->max_contour_length());
        for (auto &region_model_ptr :
             body2region_model_ptrs_map_.at(body_name)) {
          auto region_modality_ptr{std::make_shared<m3t::RegionModality>(
              region_model_ptr->body_ptr()->name() + "_region_modality" +
                  std::to_string(cam_id),
              body_ptr, color_camera_ptrs[cam_id], region_model_ptr)};
          region_modality_ptr->set_n_unoccluded_iterations(0);
          region_modality_setter_(region_modality_ptr);
          region_modality_ptr->set_reference_contour_length(max_contour_length);
          if (measure_occlusions_region_)
            region_modality_ptr->MeasureOcclusions(depth_camera_ptrs[cam_id]);
          if (model_occlusions_region_) {
             region_modality_ptr->ModelOcclusions(color_depth_renderer_ptrs[cam_id]);
            
          }
          region_modality_ptr->SetUp();
          link_ptr->AddModality(region_modality_ptr);
          cam_link_ptrs[cam_id][obj_id].push_back(region_modality_ptr);
        }
      }
      
      // Init depth modality
      if (use_depth_modality_) {
        auto depth_modality_ptr{std::make_shared<m3t::DepthModality>(
            body_name + "_depth_modality" + std::to_string(cam_id), body_ptr,
            depth_camera_ptrs[cam_id],
            body2depth_model_ptr_map_.at(body_name))};
        depth_modality_ptr->set_n_unoccluded_iterations(0);
        depth_modality_setter_(depth_modality_ptr);
        if (measure_occlusions_depth_) depth_modality_ptr->MeasureOcclusions();
        if (model_occlusions_depth_) {
          
            depth_modality_ptr->ModelOcclusions(depth_depth_renderer_ptrs[cam_id]);
          
        }
        //std::cout << depth_depth_renderer_ptrs.size();
        depth_modality_ptr->SetUp();
        link_ptr->AddModality(depth_modality_ptr);
        cam_link_ptrs[cam_id][obj_id].push_back(depth_modality_ptr);
        
      }
      

      // Init texture modality
      if (use_texture_modality_) {

          auto texture_modality_ptr{std::make_shared<m3t::TextureModality>(
              body_name + "_texture_modality" + std::to_string(cam_id),
              body_ptr, color_camera_ptrs[cam_id],
              color_silhouette_renderer_ptrs[cam_id])};
          texture_modality_setter_(texture_modality_ptr);
          if (measure_occlusions_texture_)
            texture_modality_ptr->MeasureOcclusions(depth_camera_ptrs[cam_id]);
          texture_modality_ptr->SetUp();
          link_ptr->AddModality(texture_modality_ptr);
          cam_link_ptrs[cam_id][obj_id].push_back(texture_modality_ptr);
        
      }

      
    }
      // Init optimizer
      link_ptr->SetUp();
      link_ptrs.push_back(link_ptr);
      auto optimizer_ptr{
          std::make_shared<m3t::Optimizer>(body_name + "_optimizer", link_ptr)};
      optimizer_setter_(optimizer_ptr);
      optimizer_ptr->SetUp();
      (*tracker_ptr)->AddOptimizer(optimizer_ptr);
      
      obj_id++;
    }
    
    // Init tracker
    tracker_setter_(*tracker_ptr);
    return (*tracker_ptr)->SetUp(false);
  }

void TLESSEvaluator::ResetBodies(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
      const std::map<int, m3t::Transform3fA> body2world_poses_map) const {
    int idx = 0;
  for (const auto &[body_id, obj_pose] : body2world_poses_map) {
      for (auto &body_ptr : tracker_ptr->body_ptrs()) {
        if (body_ptr->name() == tracked_body_names_[body_id])
          body_ptr->set_body2world_pose(obj_pose);
      }
      idx++;
  }
}

void TLESSEvaluator::GetBodyPoses(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    std::vector<m3t::Transform3fA> &body2world_poses) {
  body2world_poses.resize(tracked_body_names_.size());
  m3t::Transform3fA empty;
  empty.matrix() = Eigen::Matrix4f::Zero();
  int idx = 0;
  for (const auto &body_name : tracked_body_names_) {
    const auto &body_ptr = *std::find_if(
        begin(tracker_ptr->body_ptrs()), end(tracker_ptr->body_ptrs()),
        [&](const auto &b) { return b->name() == body_name; });
    if (body_ptr->body2world_pose().matrix().array().isFinite().all())
      body2world_poses[idx] = body_ptr->body2world_pose();
    else
      body2world_poses[idx] = empty;
    idx++;
  }
}

Eigen::Matrix<float, 6, 6> Adjoint(const m3t::Transform3fA &pose) {
  Eigen::Matrix<float, 6, 6> m{Eigen::Matrix<float, 6, 6>::Zero()};
  m.topLeftCorner<3, 3>() = pose.rotation();
  m.bottomLeftCorner<3, 3>() =
      m3t::Vector2Skewsymmetric(pose.translation()) * pose.rotation();
  m.bottomRightCorner<3, 3>() = pose.rotation();
  return m;
}

void TLESSEvaluator::ExecuteMeasuredTrackingCycle(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    std::vector<std::shared_ptr<m3t::LoaderColorCamera>> &color_camera_ptrs,
    std::vector<std::shared_ptr<m3t::LoaderDepthCamera>> &depth_camera_ptrs,
    std::vector<std::shared_ptr<m3t::Link>> link_ptrs,
    std::vector<std::vector<std::vector<std::shared_ptr<m3t::Modality>>>>
        cam_link_ptr,
    std::vector<int> views_initialized, std::vector<int> inlier_labels,
    int iteration) {
  // Update Cameras
  //tracker_ptr->UpdateCameras(iteration);

  for (int corr_iteration = 0;
       corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {
    // Calculate correspondences

    tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);
    // Visualize correspondences
    int corr_save_idx =
        iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
    tracker_ptr->VisualizeCorrespondences(corr_save_idx);

    // std::cout << tracker_ptr->n_corr_iterations() << std::endl;
    for (int update_iteration = 0;
         update_iteration < tracker_ptr->n_corr_iterations();
         ++update_iteration) {
      // Calculate gradient and hessian
      tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration, update_iteration);
      
      
      // Optimize camera poses
      
      for (int cam_id = 1; cam_id < color_camera_ptrs.size(); cam_id++) {
        
        Eigen::VectorXf b{Eigen::VectorXf::Zero(6)};
        Eigen::MatrixXf a{Eigen::MatrixXf::Zero(6, 6)};
        Eigen::Matrix<float, 6, 1> gradient{};
        Eigen::Matrix<float, 6, 6> hessian{};
        int valid_obj_size = 0;
        for (int obj_id = 0; obj_id < body_name2idx_map_.size(); obj_id++) {
          gradient.setZero();
          hessian.setZero();

          if (body_ptrs[obj_id]->body2world_pose().translation()(0) != 0 &&
              body_ptrs[obj_id]->body2world_pose().translation()(1) != 0 &&
              body_ptrs[obj_id]->body2world_pose().translation()(2) != 0 &&
              std::count(inlier_labels.begin(), inlier_labels.end(), obj_id) >
                  0) {
             m3t::Transform3fA cam2obj_pose =
                body_ptrs[obj_id]->body2world_pose().inverse() *
                color_camera_ptrs[cam_id]->camera2world_pose();
             auto jacobian{Adjoint(cam2obj_pose)};
             for (auto &modality_ptr : cam_link_ptr[cam_id][obj_id]) {
               gradient += modality_ptr->gradient();
               hessian += modality_ptr->hessian();
               //std::cout << modality_ptr->gradient().transpose() << std::endl;
               
             }
             
              auto b_update = jacobian.transpose() * gradient;
             if (b_update.array().isFinite().all() &&
                 std::count(views_initialized.begin(), views_initialized.end(),
                            cam_id) >0) {
                 valid_obj_size += 1;
                 b += b_update;
                 a.triangularView<Eigen::Lower>() -=
                  jacobian.transpose() * hessian * jacobian;
             }
             
             
             for (auto &modality_ptr : cam_link_ptr[cam_id][obj_id]) {
               //if (corr_iteration < 2)
               //    modality_ptr->SetGradientAndHessian(
               //    0.0 * modality_ptr->gradient(), 0.0 * modality_ptr->hessian());
               
               //else
                 modality_ptr->SetGradientAndHessian(
                     0.1 * modality_ptr->gradient(),
                     0.1 * modality_ptr->hessian());

               if (std::count(views_initialized.begin(),
                              views_initialized.end(), cam_id) <1)
                 modality_ptr->SetGradientAndHessian(
                     0.0 * modality_ptr->gradient(),
                     0.0 * modality_ptr->hessian());
             }
             
             
             
          }

        }
        b = 0.05*b / valid_obj_size;
        a = a / valid_obj_size;
        
        //calculate average gradient and hessian
        

        
       
        Eigen::LDLT<Eigen::MatrixXf, Eigen::Lower> ldlt{a};
        Eigen::VectorXf theta{ldlt.solve(b)};
        // std::cout << theta.transpose() << std::endl;
        if (theta.array().isNaN().isZero()) {
          // Calculate pose variation
          m3t::Transform3fA pose_variation{m3t::Transform3fA::Identity()};
          pose_variation.translate(-theta.tail<3>());
          pose_variation.rotate(
              m3t::Vector2Skewsymmetric(-theta.head<3>()).exp());
          
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
            depth_pose.matrix() =
                color_camera_ptrs[cam_id]->camera2world_pose().matrix();
            depth_camera_ptrs[cam_id]->set_camera2world_pose(depth_pose);
          }
          
        } 
      }
      
      tracker_ptr->CalculateOptimization(iteration, corr_iteration, update_iteration);

      // Visualize pose update
      int update_save_idx =
          corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
      tracker_ptr->VisualizeOptimization(update_save_idx);
    }
  }

  // Calculate results
  tracker_ptr->CalculateResults(iteration);

  // Visualize results and update viewers
  tracker_ptr->VisualizeResults(iteration);
}

void TLESSEvaluator::ExecuteMeasuredRefinementCycle(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    int cam_id,
    int iteration) const {
  auto refiner_ptr{tracker_ptr->refiner_ptrs()[cam_id]};
  for (int corr_iteration = 0;
       corr_iteration < refiner_ptr->n_corr_iterations(); ++corr_iteration) {

    // Calculate correspondences
    refiner_ptr->CalculateCorrespondences(corr_iteration);

    // Visualize correspondences
    int corr_save_idx = corr_iteration;
    refiner_ptr->VisualizeCorrespondences(corr_save_idx);

    for (int update_iteration = 0;
         update_iteration < 4;
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



void TLESSEvaluator::UpdateCameras(std::vector<int> load_indices) const {
  for (int cam_id = 0; cam_id < num_view_; ++cam_id) {
        //std::cout << load_indices[cam_id] << std::endl;
        if (use_region_modality_ || use_texture_modality_) {
          auto color_camera{std::static_pointer_cast<m3t::LoaderColorCamera>(
              color_camera_ptrs[cam_id])};
          color_camera->set_load_index(load_indices[cam_id]);
          color_camera->SetUp();
        }
        //std::cout << "color camera" << std::endl;
        if (use_depth_modality_) {
          auto depth_camera{std::static_pointer_cast<m3t::LoaderDepthCamera>(
              depth_camera_ptrs[cam_id])};
          depth_camera->set_load_index(load_indices[cam_id]);
          depth_camera->SetUp();
        }
        //std::cout << "depth camera" << std::endl;
      
  }

  
}

void TLESSEvaluator::Posedetector(
    mmdeploy_pose_detector_t pose_detector, std::vector<cv::Mat> images,
    std::vector<cv::Matx33d> cam_Ks,
    //std::vector<std::vector<m3t::Transform3fA>> &body2world_poses_all,
    cosypose::Ransac_candidates &candidates,
    int num_id) {
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
  int status = mmdeploy_pose_detector_apply(pose_detector, mats.data(),
                                            batch_size, &res);

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

      kpts_2d.clear();
      for (int k = 0; k < num_point; k++) {
        kpts_2d.push_back(
            cv::Point2f((int)res_ptr->point[i * num_point + k].x,
                        (int)res_ptr->point[i * num_point + k].y));
      }
      cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
      cv::Mat rvec, tvec;

      bool solution;
      solution = cv::solvePnPRansac(kpts_gt_[obj_id], kpts_2d, cam_Ks[img_i],
                                    distCoeffs, rvec, tvec, false);

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

static double random() { return double(rand()) / (double(RAND_MAX) + 1.0); }
  // return int in range [a,b)
static int random_int(int a, int b) {
  double u = floor(random() * (b - a) + a);
  return static_cast<int>(u);
}


TLESSEvaluator::Result TLESSEvaluator::CalculateAverageBodyResult(
    const std::vector<std::string> &body_names) const {
  size_t n_results = 0;
  std::vector<Result> result_sums;
  for (const auto &result : results_) {
    if (std::find(begin(body_names), end(body_names), result.body_name) !=
        end(body_names)) {
      n_results += result.frame_results.size();
      result_sums.push_back(SumResults(result.frame_results));
    }
  }
  return DivideResult(SumResults(result_sums), n_results);
}

TLESSEvaluator::Result TLESSEvaluator::CalculateAverageSequenceResult(
    const std::vector<std::string> &sequence_names) const {
  size_t n_results = 0;
  std::vector<Result> result_sums;
  for (const auto &result : results_) {
    if (std::find(begin(sequence_names), end(sequence_names),
                  result.sequence_name) != end(sequence_names)) {
      n_results += result.frame_results.size();
      result_sums.push_back(SumResults(result.frame_results));
    }
  }
  return DivideResult(SumResults(result_sums), n_results);
}

TLESSEvaluator::Result TLESSEvaluator::CalculateAverageResult(
    const std::vector<Result> &results) {
  return DivideResult(SumResults(results), results.size());
}

TLESSEvaluator::Result TLESSEvaluator::SumResults(
    const std::vector<Result> &results) {
  Result sum;
  for (const auto &result : results) {
    sum.add_auc += result.add_auc;
    sum.adds_auc += result.adds_auc;
    std::transform(begin(result.add_curve), end(result.add_curve),
                   begin(sum.add_curve), begin(sum.add_curve),
                   [](float a, float b) { return a + b; });
    std::transform(begin(result.adds_curve), end(result.adds_curve),
                   begin(sum.adds_curve), begin(sum.adds_curve),
                   [](float a, float b) { return a + b; });
    sum.execution_times.pose_estimation +=
        result.execution_times.pose_estimation;
    sum.execution_times.candidate_matching +=
        result.execution_times.candidate_matching;
    sum.execution_times.bundle_adjustment +=
        result.execution_times.bundle_adjustment;
    sum.execution_times.complete_cycle += result.execution_times.complete_cycle;
  }
  return sum;
}

TLESSEvaluator::Result TLESSEvaluator::DivideResult(const Result &result,
                                                size_t n) {
  Result divided;
  divided.add_auc = result.add_auc / float(n);
  divided.adds_auc = result.adds_auc / float(n);
  std::transform(begin(result.add_curve), end(result.add_curve),
                 begin(divided.add_curve),
                 [&](float a) { return a / float(n); });
  std::transform(begin(result.adds_curve), end(result.adds_curve),
                 begin(divided.adds_curve),
                 [&](float a) { return a / float(n); });
  divided.execution_times.pose_estimation =
      result.execution_times.pose_estimation / float(n);
  divided.execution_times.candidate_matching =
      result.execution_times.candidate_matching / float(n);
  divided.execution_times.bundle_adjustment =
      result.execution_times.bundle_adjustment / float(n);
  divided.execution_times.complete_cycle =
      result.execution_times.complete_cycle / float(n);
  return divided;
}

void TLESSEvaluator::CalculatePoseResults(
    const std::shared_ptr<m3t::Tracker> tracker_ptr,
    const std::string &body_name, const m3t::Transform3fA &gt_body2world_pose,
    Result *result) const {
  const auto &vertices{body2reduced_vertice_map_.at(body_name)};
  const auto &kdtree_index{body2kdtree_ptr_map_.at(body_name)->index};

  // Calculate pose error
  m3t::Transform3fA body2world_pose;
  for (const auto &body_ptr : tracker_ptr->body_ptrs()) {
    if (body_ptr->name() == body_name)
      body2world_pose = body_ptr->body2world_pose();
  }
  m3t::Transform3fA delta_pose{body2world_pose.inverse() * gt_body2world_pose};

  // Calculate add and adds error
  float add_error = 0.0f;
  float adds_error = 0.0f;
  size_t ret_index;
  float dist_sqrt;
  Eigen::Vector3f v;
  for (const auto &vertice : vertices) {
    v = delta_pose * vertice;
    add_error += (vertice - v).norm();
    kdtree_index->knnSearch(v.data(), 1, &ret_index, &dist_sqrt);
    adds_error += std::sqrt(dist_sqrt);
  }
  add_error /= vertices.size();
  adds_error /= vertices.size();

  // Calculate curve (tracking loss distribution)
  std::fill(begin(result->add_curve), end(result->add_curve), 1.0f);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    if (add_error < thresholds_[i]) break;
    result->add_curve[i] = 0.0f;
  }
  std::fill(begin(result->adds_curve), end(result->adds_curve), 1.0f);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    if (adds_error < thresholds_[i]) break;
    result->adds_curve[i] = 0.0f;
  }

  // Calculate area under curve
  result->add_auc = 1.0f - std::min(add_error / kThresholdMax, 1.0f);
  result->adds_auc = 1.0f - std::min(adds_error / kThresholdMax, 1.0f);
}

bool TLESSEvaluator::LoadGTPoses(
    const std::string &sequence_name,
    std::vector <std::vector<m3t::Transform3fA>> * gt_body2world_poses,
    std::vector <std::vector<m3t::Transform3fA>> *gt_camera_poses) const {
  // Open poses file
  std::filesystem::path obj_path{dataset_directory_ / "test_primesense" /sequence_name /
                             "scene_gt.json"};
  std::filesystem::path cam_path{dataset_directory_ / "test_primesense" / sequence_name /
                                 "scene_camera.json"};
  // Open file
  cv::FileStorage fs_obj;
  cv::FileStorage fs_cam;
  
  try {
    fs_obj.open(cv::samples::findFile(obj_path.string()), cv::FileStorage::READ);
    fs_cam.open(cv::samples::findFile(cam_path.string()), cv::FileStorage::READ);
  } catch (cv::Exception e) {
  }
  if (!fs_obj.isOpened()) {
    std::cerr << "Could not open file " << obj_path << std::endl;
    return false;
  }
  if (!fs_cam.isOpened()) {
    std::cerr << "Could not open file " << cam_path << std::endl;
    return false;
  }
  
  // Iterate all sequences
  gt_body2world_poses->clear();
  gt_camera_poses->clear();
  // Define begin index, n_frames and keyframes
  
  const int n_frames = sequence2keyframes_map_.at(sequence_name).size();
  const auto keyframes = sequence2keyframes_map_.at(sequence_name);
  //std::cout << n_frames << std::endl;
  std::vector<m3t::Transform3fA> camera_poses;
  m3t::Transform3fA camera_pose;
  
  for (int idx = 0; idx < n_frames; ++idx) {
    int frame_idx = keyframes[idx][0];
    camera_poses.clear();
    for (int cam_id=0; cam_id < keyframes[idx].size(); cam_id++) {
      cv::FileNode fn_sequence_cam{fs_cam[std::to_string(keyframes[idx][cam_id])]};
     
      for (int i = 0; i < 9; ++i)
        fn_sequence_cam["cam_R_w2c"][i] >> camera_pose.matrix()(i / 3, i % 3);
      for (int i = 0; i < 3; ++i)
        fn_sequence_cam["cam_t_w2c"][i] >> camera_pose.matrix()(i, 3);
      camera_pose.translation() *= 0.001;
      camera_poses.push_back(camera_pose.inverse());
    }
    
    gt_camera_poses->push_back(camera_poses);
    
    cv::FileNode fn_sequence_obj{fs_obj[std::to_string(frame_idx)]};
    if (fn_sequence_obj.empty()) break;

    // Iterate all bodies
    std::vector<m3t::Transform3fA> body2world_poses(body_name2idx_map_.size());
    float execution_time;
    for (const auto &[body_name, body_idx] : body_name2idx_map_) {
      int body_id = std::stoi(body_name.substr(4, body_name.size()));
      // Find file node for body
      cv::FileNode fn_body;
      for (const auto &fn_obj : fn_sequence_obj) {
        int obj_id;
        fn_obj["obj_id"] >> obj_id;
        if (obj_id == body_id) {
          fn_body = fn_obj;
          break;
        }
      }

      // Extract object poses
      m3t::Transform3fA body2world_pose;
      if (fn_body.empty()) {
        //std::cerr << "obj_id " << body_id << " not found in sequence " << frame_idx
        //          << std::endl;
        //return false;
        memset(&body2world_pose, 0, sizeof(body2world_pose));
      } else {
        for (int i = 0; i < 9; ++i)
          fn_body["cam_R_m2c"][i] >> body2world_pose.matrix()(i / 3, i % 3);
        for (int i = 0; i < 3; ++i)
          fn_body["cam_t_m2c"][i] >> body2world_pose.matrix()(i, 3);
        body2world_pose.translation() *= 0.001;
      }
      
     
      body2world_poses[body_idx-1] = body2world_pose;
    }
    gt_body2world_poses->push_back(std::move(body2world_poses));
  }
  return true;
}


bool TLESSEvaluator::LoadDetectorPoses(
    const std::string &sequence_name, const std::string &body_name,
    std::vector<m3t::Transform3fA> *detector_body2world_poses,
    std::vector<bool> *body_detected) const {
  // Open detector poses file
  std::filesystem::path path{external_directory_ / "poses" / detector_folder_ /
                             (sequence_name + "_" + body_name + ".txt")};
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream " << path.string() << std::endl;
    return false;
  }

  // Read poses
  body_detected->clear();
  detector_body2world_poses->clear();
  std::string parsed;
  while (std::getline(ifs, parsed)) {
    std::stringstream ss{parsed};
    Eigen::Quaternionf quaternion;
    Eigen::Translation3f translation;
    std::getline(ss, parsed, ' ');
    quaternion.w() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.x() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.y() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.z() = stof(parsed);
    std::getline(ss, parsed, ' ');
    translation.x() = stof(parsed);
    std::getline(ss, parsed, ' ');
    translation.y() = stof(parsed);
    std::getline(ss, parsed);
    translation.z() = stof(parsed);
    if (translation.vector().isZero()) {
      detector_body2world_poses->push_back(m3t::Transform3fA::Identity());
      body_detected->push_back(false);
    } else {
      quaternion.normalize();
      detector_body2world_poses->push_back(
          m3t::Transform3fA{translation * quaternion});
      body_detected->push_back(true);
    }
  }
  return true;
}

void TLESSEvaluator::VisualizeResult(const Result &result,
                                   const std::string &title) {
  std::cout << title << ": ";
  if (result.frame_index) std::cout << "frame " << result.frame_index << ": ";
  std::cout << "pose_estimation_time = " << result.execution_times.pose_estimation
            << " us, "
            << "candidate_matching_time = " << result.execution_times.candidate_matching
            << " us, "
            << "bundle_adjustment_time = " << result.execution_times.bundle_adjustment
            << " us, "
            << "execution_time = " << result.execution_times.complete_cycle
            << " us, " << std::endl;
}

bool TLESSEvaluator::CreateRunConfigurations() {
  run_configurations_.clear();
  for (int sequence_id : sequence_ids_) {
        RunConfiguration run_configuration;
        run_configuration.sequence_id = sequence_id;
        run_configuration.sequence_name = SequenceIDToName(sequence_id);
        for (const auto &body_name : evaluated_body_names_) {
            run_configuration.evaluated_body_names.push_back(body_name);
        }
        if (run_configuration.evaluated_body_names.empty()) continue;
      
        run_configuration.tracked_body_names =
            run_configuration.evaluated_body_names;
      
        run_configurations_.push_back(std::move(run_configuration));
    
    
  }
  return true;
}

void TLESSEvaluator::AssembleTrackedBodyNames() {
  tracked_body_names_.clear();
  for (const auto &run_configuration : run_configurations_) {
    for (const auto &tracked_body_name : run_configuration.tracked_body_names) {
      if (std::find(begin(tracked_body_names_), end(tracked_body_names_),
                    tracked_body_name) == end(tracked_body_names_))
        tracked_body_names_.push_back(tracked_body_name);
    }
  }
}

bool TLESSEvaluator::LoadBodies() {
  bool success = true;
  body2body_ptr_map_.clear();
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    // Set up main bodies
    auto &body_name{tracked_body_names_[i]};
    std::filesystem::path directory{dataset_directory_ / "models" /
                                    (body_name + ".obj")};
    auto body_ptr{
        std::make_shared<m3t::Body>(body_name, directory, 0.001f,
                                    true, true, m3t::Transform3fA::Identity())};
    body_ptr->set_body_id(10 + 10 * i);
    if (!body_ptr->SetUp()) {
      success = false;
      continue;
    }
#pragma omp critical
    body2body_ptr_map_.insert({body_name, std::move(body_ptr)});

    // Set up sub bodies if required
    if (use_multi_region_ &&
        std::find(begin(multi_region_body_names_),
                  end(multi_region_body_names_),
                  body_name) != end(multi_region_body_names_)) {
      int id_counter = 0;
      std::vector<std::shared_ptr<m3t::Body>> sub_body_ptrs{};
      std::filesystem::path multi_region_directory{external_directory_ /
                                                   "multi_region" / body_name};
      for (auto &file :
           std::filesystem::directory_iterator(multi_region_directory)) {
        auto sub_body_ptr{std::make_shared<m3t::Body>(
            file.path().stem().string(), file.path(), 0.001f, true, true,
            m3t::Transform3fA::Identity())};
        sub_body_ptr->set_body_id(10 + 10 * i + id_counter++);
        if (!sub_body_ptr->SetUp()) {
          success = false;
          continue;
        }
        sub_body_ptrs.push_back(std::move(sub_body_ptr));
      }
      body2sub_body_ptrs_map_.insert({body_name, std::move(sub_body_ptrs)});
    }
  }
  return success;
}

bool TLESSEvaluator::GenerateModels() {
  std::filesystem::path directory{external_directory_ / "models"};
  std::filesystem::create_directories(directory);
  for (auto body_name : tracked_body_names_) {
    if (use_region_modality_) {
      std::vector<std::shared_ptr<m3t::RegionModel>> region_model_ptrs{};
      if (use_multi_region_ &&
          std::find(begin(multi_region_body_names_),
                    end(multi_region_body_names_),
                    body_name) != end(multi_region_body_names_)) {
        for (auto &body_ptr : body2sub_body_ptrs_map_.at(body_name)) {
          auto model_ptr{std::make_shared<m3t::RegionModel>(
              body_ptr->name() + "_region_model", body_ptr,
              directory / (body_ptr->name() + "_region_model.bin"), 0.8f, 4,
              500, 0.05f, 0.002f, false, 2000)};
          for (auto &sub_body : body2sub_body_ptrs_map_.at(body_name)) {
            if (sub_body->name() != body_ptr->name())
              model_ptr->AddAssociatedBody(sub_body, false, false);
          }
          region_model_setter_(model_ptr);
          if (!model_ptr->SetUp()) return false;
          region_model_ptrs.push_back(std::move(model_ptr));
        }
      } else {
        auto model_ptr{std::make_shared<m3t::RegionModel>(
            body_name + "_region_model", body2body_ptr_map_[body_name],
            directory / (body_name + "_region_model.bin"), 0.8f, 4, 500, 0.05f,
            0.002f, false, 2000)};
        region_model_setter_(model_ptr);
        if (!model_ptr->SetUp()) return false;
        region_model_ptrs.push_back(std::move(model_ptr));
      }
      body2region_model_ptrs_map_.insert(
          {body_name, std::move(region_model_ptrs)});
    }
    if (use_depth_modality_) {
      auto model_ptr{std::make_shared<m3t::DepthModel>(
          body_name + "_depth_model", body2body_ptr_map_[body_name],
          directory / (body_name + "_depth_model.bin"), 0.8f, 4, 500, 0.05f,
          0.002f, false, 2000)};
      depth_model_setter_(model_ptr);
      if (!model_ptr->SetUp()) return false;
      body2depth_model_ptr_map_.insert({body_name, std::move(model_ptr)});
    }
  }
  return true;
}

bool TLESSEvaluator::LoadKeyframes() {
  bool success = true;
  sequence2keyframes_map_.clear();

    cv::FileStorage fs;

    // Open poses file
    std::filesystem::path path{dataset_directory_ / "evaluate_multiview" /
                               ("view" + std::to_string(num_view_) + ".json")};
    try {
      fs.open(cv::samples::findFile(path.string()), cv::FileStorage::READ);
    } catch (cv::Exception e) {
    }
    cv::FileNode scene_sequence = fs["scene_ids"];
    std::vector<int> scene_ids;
    int scene_id;
    int scene_id_last = scene_sequence[0];
    cv::FileNode view_sequence = fs["view_ids"];
    std::vector<std::vector<int>> view_ids;
    std::vector<int> view_id;
    int last_frame;
    for (last_frame = view_sequence.size() - 1; last_frame > 0; --last_frame) {
      if (view_sequence[last_frame].size() == num_view_) break;
    }
    for (int num_frame = 0; num_frame < view_sequence.size(); ++num_frame) {
      scene_id = scene_sequence[num_frame];
      view_id.clear();
      if (view_sequence[num_frame].size() != num_view_)
        continue;
      for (int num_view = 0; num_view < view_sequence[num_frame].size();
           ++num_view)
        view_id.push_back(view_sequence[num_frame][num_view]);

      if (scene_id == scene_id_last) {
        view_ids.push_back(view_id);
      } else {
        int n_zeros = std::max(6 - int(std::to_string(scene_id_last).length()), 0);
        sequence2keyframes_map_.insert({std::string(n_zeros, '0') +
                                            std::to_string(scene_id_last),
                                        std::move(view_ids)});
        view_ids.clear();
        view_ids.push_back(view_id);
      }
      if (num_frame ==last_frame) {
        int n_zeros = std::max(6 - int(std::to_string(scene_id).length()), 0);
        sequence2keyframes_map_.insert({std::string(n_zeros, '0')
                                            +std::to_string(scene_id),
                                        std::move(view_ids)});
      }
        
      scene_id_last = scene_id;
    }
  
  return success;
}



void TLESSEvaluator::GenderateReducedVertices() {
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    auto &body_name{tracked_body_names_[i]};
    const auto &vertices{body2body_ptr_map_[body_name]->vertices()};
    if (n_vertices_evaluation_ <= 0 ||
        n_vertices_evaluation_ >= vertices.size()) {
#pragma omp critical
      body2reduced_vertice_map_[body_name] = vertices;
      continue;
    }

    std::mt19937 generator{7};
    if (use_random_seed_)
      generator.seed(unsigned(
          std::chrono::system_clock::now().time_since_epoch().count()));

    std::vector<Eigen::Vector3f> reduced_vertices(n_vertices_evaluation_);
    int n_vertices = vertices.size();
    for (auto &v : reduced_vertices) {
      int idx = int(generator() % n_vertices);
      v = vertices[idx];
    }
#pragma omp critical
    body2reduced_vertice_map_.insert({body_name, std::move(reduced_vertices)});
  }
}

void TLESSEvaluator::GenerateKDTrees() {
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    auto &body_name{tracked_body_names_[i]};
    const auto &vertices{body2body_ptr_map_[body_name]->vertices()};
    auto kdtree_ptr{std::make_unique<KDTreeVector3f>(3, vertices, 10)};
    kdtree_ptr->index->buildIndex();
#pragma omp critical
    body2kdtree_ptr_map_.insert({body_name, std::move(kdtree_ptr)});
  }
}




std::string TLESSEvaluator::SequenceIDToName(int sequence_id) const {
  int n_zeros = 6 - int(std::to_string(sequence_id).length());
  return std::string(n_zeros, '0') + std::to_string(sequence_id);
}

float TLESSEvaluator::ElapsedTime(
    const std::chrono::high_resolution_clock::time_point &begin_time) {
  auto end_time{std::chrono::high_resolution_clock::now()};
  return float(std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     begin_time)
                   .count());
}
