// Copyright (c) HEX lab, Technical University of Munich

#include "evaluate_tless_evaluator_multiview.h"

int main() {
  // Directories
  std::filesystem::path model_directory{"path/to/checkpoint_tless"};
  std::filesystem::path dataset_directory{"path/to/dataset/tless"};
  std::filesystem::path external_directory{"path/to/dataset/tless/external"};
  std::filesystem::path result_path{"path/to/dataset/tless/result"};

  // Dataset configuration
  //std::vector<int> sequence_ids{58, 59};
  std::vector<int> sequence_ids{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
      20};
  std::vector<std::string> body_names{"obj_000001", "obj_000002", "obj_000003", "obj_000004", "obj_000005", "obj_000006", "obj_000007", "obj_000008", "obj_000009", "obj_000010",
                                      "obj_000011", "obj_000012", "obj_000013", "obj_000014", "obj_000015", "obj_000016", "obj_000017", "obj_000018", "obj_000019", "obj_000020",
                                      "obj_000021", "obj_000022", "obj_000023", "obj_000024", "obj_000025", "obj_000026", "obj_000027", "obj_000028", "obj_000029", "obj_000030"};
  int num_view = 4;

  // Run experiments
  TLESSEvaluator evaluator{num_view, "evaluator", model_directory, dataset_directory,
                         external_directory, sequence_ids,
                         body_names};
  evaluator.set_region_modality_setter([&](auto r) {
    r->set_n_lines_max(200);
    r->set_use_adaptive_coverage(false);
    r->set_min_continuous_distance(3.0f);
    r->set_function_length(8);
    r->set_distribution_length(12);
    r->set_function_amplitude(0.43f);
    r->set_function_slope(0.5f);
    r->set_learning_rate(1.3f);
    r->set_scales({7, 4, 2}); // {5, 5, 3}
    r->set_standard_deviations({25.0f, 15.0f, 10.0f}); // {20.0f, 10.0f, 10.0f}
    r->set_n_histogram_bins(16);
    r->set_learning_rate_f(0.2f);
    r->set_learning_rate_b(0.2f);
    r->set_unconsidered_line_length(0.5f);
    r->set_max_considered_line_length(20.0f);
    r->set_measured_depth_offset_radius(0.01f);
    r->set_measured_occlusion_radius(0.01f);
    r->set_measured_occlusion_threshold(0.03f);
  });
  evaluator.set_depth_modality_setter([&](auto d) {
    d->set_n_points_max(200);
    d->set_use_adaptive_coverage(false);
    d->set_use_depth_scaling(false);
    d->set_stride_length(0.005f); // 0.01f
    d->set_considered_distances({0.07f, 0.05f, 0.04f}); // {0.3f, 0.3f, 0.1f}
    d->set_standard_deviations({0.05f, 0.03f, 0.02f}); // {0.3, 0.1f, 0.025f}
    d->set_measured_depth_offset_radius(0.01f);
    d->set_measured_occlusion_radius(0.01f);
    d->set_measured_occlusion_threshold(0.03f);
  });
  evaluator.set_texture_modality_setter([&](auto t) {
    t->set_descriptor_type(m3t::TextureModality::DescriptorType::ORB);
    t->set_focused_image_size(200);
    t->set_descriptor_distance_threshold(0.7f);
    t->set_tukey_norm_constant(20.0f);
    t->set_standard_deviations({10.0f, 10.0f, 3.0f});
    t->set_max_keyframe_rotation_difference(10.0f * m3t::kPi / 180.0f);
    t->set_max_keyframe_age(1000);
    t->set_n_keyframes(1);
    t->set_orb_n_features(300);
    t->set_orb_scale_factor(1.2f);
    t->set_orb_n_levels(3);
    t->set_brisk_threshold(35);
    t->set_brisk_octave(3);
    t->set_brisk_pattern_scale(0.6f);
    t->set_daisy_radius(8.0f);
    t->set_daisy_q_radius(3);
    t->set_daisy_q_theta(4);
    t->set_daisy_q_hist(8);
    t->set_freak_orientation_normalized(true);
    t->set_freak_scale_normalized(true);
    t->set_freak_pattern_scale(16.0f);
    t->set_freak_n_octaves(4);
    t->set_sift_n_features(0);
    t->set_sift_n_octave_layers(3);
    t->set_sift_contrast_threshold(0.04);
    t->set_sift_edge_threshold(10.0);
    t->set_sift_sigma(0.7f);
    t->set_measured_occlusion_radius(0.01f);
    t->set_measured_occlusion_threshold(0.03f);
  });
  evaluator.set_optimizer_setter([&](auto o) {
    o->set_tikhonov_parameter_rotation(100.0f);
    o->set_tikhonov_parameter_translation(1000.0f); // 1000.0f
  });
  evaluator.set_tracker_setter([&](auto t) {
    t->set_n_update_iterations(2);
    t->set_n_corr_iterations(4); // 7
  });
  evaluator.set_evaluate_refinement(false);
  evaluator.set_detector_folder("cosypose");
  evaluator.set_use_matlab_gt_poses(false);
  evaluator.set_run_sequentially(true);
  evaluator.set_use_random_seed(false);
  evaluator.set_n_vertices_evaluation(1000);
  evaluator.set_visualize_frame_results(false);
  evaluator.set_visualize_tracking(false);
  evaluator.set_use_multi_region(true);
  evaluator.set_use_region_modality(true);
  evaluator.set_use_depth_modality(true);
  evaluator.set_use_texture_modality(false);
  evaluator.set_measure_occlusions_region(true);
  evaluator.set_measure_occlusions_depth(true);
  evaluator.set_measure_occlusions_texture(true);
  evaluator.set_model_occlusions_region(true);
  evaluator.set_model_occlusions_depth(true);
  evaluator.set_model_occlusions_texture(true);
  evaluator.SetUp();
  evaluator.Evaluate();
  evaluator.SaveResults(result_path);
  return 0;
}
