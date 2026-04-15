#ifndef _COSYPOSE_HPP
#define _COSYPOSE_HPP

#include <random>
#include <tuple>
#include <unordered_map>
#include <map>
#include <set>
#include <numeric>
#include <Eigen/Core>
#include <torch/torch.h>
#include <stdexcept>



using namespace torch::indexing;

namespace cosypose {

struct Match {
  int c1, c2;
};

struct Symmetry_info {
  std::vector<int> ids_expand;
  std::vector<int> labels_expand;
  std::vector<int> sym_ids;
};

struct Mesh_database {
  std::unordered_map<int, int> n_sym_mapping;
  // std::vector<std::vector<m3t::Transform3fA>> symmetries_;
  torch::Tensor sym_tensor;  // obj_id, sym, 4, 4
  torch::Tensor kpts_3d;     // obj_id, n_kpts, 3
};

struct Ransac_inlier {
  std::vector<int> inlier_c1;

  std::vector<int> inlier_c2;

  std::vector<int> best_hypo;
};

struct Ransac_candidates {
  std::vector<int> view_ids, labels;  // to initialize
  std::vector<int> view1, view2;
  std::vector<int> match1_cand1, match1_cand2, match2_cand1, match2_cand2;
  std::vector<int> hypothesis_id, cand1, cand2;
  torch::Tensor poses;  // detect_id, 4, 4, to initialize
};

struct Results {
  std::vector<int> view1, view2;
  torch::Tensor TC1C2;
};

using ViewPair = std::tuple<int, int>;
using Matches = std::vector<Match>;

static std::vector<int> sort_indexes(const std::vector<float>& v) {
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

static std::vector<int> random_permutation(int N, int seed) {
  std::vector<int> vec;
  for (int i = 0; i < N; i++) {
    vec.push_back(i);
  }
  std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
  return vec;
}

static void make_ransac_infos(Ransac_candidates& candidates,
                              int n_ransac_iter = 2000, int seed = 0) {
  // Make tentative matches.
  std::map<ViewPair, Matches> tentative_matches_per_view_pair;
  int n_cand = candidates.view_ids.size();
  // std::cout << "candidate number: " + std::to_string(n_cand) << std::endl;
  for (int n = 0; n < n_cand; n++) {
    for (int m = 0; m < n_cand; m++) {
      if (candidates.view_ids[n] < candidates.view_ids[m] &&
          candidates.labels[n] == candidates.labels[m]) {
        ViewPair view_pair(candidates.view_ids[n], candidates.view_ids[m]);
        Match tentative_match({n, m});
        tentative_matches_per_view_pair[view_pair].push_back(tentative_match);
      }
    }
  }

  // Ransac seeds
  std::vector<int> seed_view1, seed_view2;
  std::vector<int> seed_match1_cand1, seed_match1_cand2, seed_match2_cand1,
      seed_match2_cand2;
  std::vector<int> mtc_hypothesis_id, mtc_cand1, mtc_cand2;

  int n_ransac_seeds;
  n_ransac_seeds = 0;
  for (auto kv : tentative_matches_per_view_pair) {
    const Matches& tentative_matches = kv.second;
    int n_tentative_matches = tentative_matches.size();
    auto perm1 = random_permutation(n_tentative_matches, seed);
    auto perm2 = random_permutation(n_tentative_matches, seed + 1);
    int n_pairs = 0;
    // Ransac seeds
    for (int m1_id : perm1) {
      if (n_pairs >= n_ransac_iter) break;
      for (int m2_id : perm2) {
        if (n_pairs >= n_ransac_iter) break;
        if (m1_id != m2_id) {
          seed_view1.push_back(std::get<0>(kv.first));
          seed_view2.push_back(std::get<1>(kv.first));
          seed_match1_cand1.push_back(tentative_matches[m1_id].c1);
          seed_match1_cand2.push_back(tentative_matches[m1_id].c2);
          seed_match2_cand1.push_back(tentative_matches[m2_id].c1);
          seed_match2_cand2.push_back(tentative_matches[m2_id].c2);
          for (int i = 0; i < n_tentative_matches; i++) {
            mtc_hypothesis_id.push_back(n_ransac_seeds);
            mtc_cand1.push_back(tentative_matches[i].c1);
            mtc_cand2.push_back(tentative_matches[i].c2);
          }
          n_pairs++;
          n_ransac_seeds++;
        }
      }
    }
  }
  
  candidates.view1 = seed_view1;
  candidates.view2 = seed_view2;
  candidates.match1_cand1 = seed_match1_cand1;
  candidates.match1_cand2 = seed_match1_cand2;
  candidates.match2_cand1 = seed_match2_cand1;
  candidates.match2_cand2 = seed_match2_cand2;
  candidates.hypothesis_id = mtc_hypothesis_id;
  candidates.cand1 = mtc_cand1;
  candidates.cand2 = mtc_cand2;
}

static void find_ransac_inliers(const std::vector<int> seeds_view1,
                                const std::vector<int> seeds_view2,
                                const std::vector<int> mtc_hypothesis_id,
                                const std::vector<int> mtc_cand1,
                                const std::vector<int> mtc_cand2,
                                const std::vector<float> dists,
                                float dist_threshold, int n_min_inliers,
                                Ransac_inlier& candidates_outputs) {
  struct RansacHypothesis {
    int hypothesis_id = 0;
    int view1 = 0, view2 = 1;
    Matches matches_inliers;
    Matches matches_inliers_uniqs;
    std::vector<float> matches_inliers_dists;
    float dists_sum = 0.0;
    int n_inliers = 0;
  };

  

  // A) Iterate over all seeds views. Build id_to_hypothesis and
  // viewpair_to_hypothesis [pointer here].
  std::unordered_map<int, RansacHypothesis> id_to_hypothesis;
  std::map<ViewPair, std::vector<int>> viewpair_to_hypotheses_ids;
  int n_hypotheses = seeds_view1.size();
  // std::cout << "n hypo: " << n_hypotheses << std::endl;
  for (int n = 0; n < n_hypotheses; n++) {
    int v1 = seeds_view1[n];
    int v2 = seeds_view2[n];
    ViewPair view_pair(v1, v2);
    RansacHypothesis hypothesis;
    hypothesis.hypothesis_id = n;
    hypothesis.view1 = v1;
    hypothesis.view2 = v2;
    hypothesis.dists_sum = 0.f;
    hypothesis.n_inliers = 0;
    hypothesis.matches_inliers_dists.clear();
    hypothesis.matches_inliers.clear();
    hypothesis.matches_inliers_uniqs.clear();
    id_to_hypothesis[n] = hypothesis;
    viewpair_to_hypotheses_ids[view_pair].push_back(n);
  }

  // B) Iterate over mtc. Fill the hypotheses scores etc.
  int n_mtc = mtc_hypothesis_id.size();
  for (int n = 0; n < n_mtc; n++) {
    RansacHypothesis& hypothesis = id_to_hypothesis[mtc_hypothesis_id[n]];
    const float& dist = dists[n];
    // std::cout << "dist: " << dist << std::endl;
    if (dist <= dist_threshold) {
      Match match({mtc_cand1[n], mtc_cand2[n]});
      hypothesis.matches_inliers.push_back(match);
      hypothesis.matches_inliers_dists.push_back(dist);
    }
  }

  // C) Find the best pairs for each hypothesis.
  for (auto kv : viewpair_to_hypotheses_ids) {
    for (auto hypothesis_id : kv.second) {
      RansacHypothesis& hypothesis = id_to_hypothesis[hypothesis_id];
      std::set<int> cand1_matched, cand2_matched;
      for (auto i : sort_indexes(hypothesis.matches_inliers_dists)) {
        Match& match = hypothesis.matches_inliers[i];
        if ((cand1_matched.count(match.c1) == 0) &&
            (cand2_matched.count(match.c2) == 0)) {
          cand1_matched.insert(match.c1);
          cand2_matched.insert(match.c2);
          hypothesis.matches_inliers_uniqs.push_back(match);
          hypothesis.dists_sum += hypothesis.matches_inliers_dists[i];
          hypothesis.n_inliers += 1;
        }
      }
    }
  }

  // D) Find the best hypotheses.
  std::vector<int> inlier_matches_cand1, inlier_matches_cand2, best_hypotheses;
  for (auto kv : viewpair_to_hypotheses_ids) {
    RansacHypothesis best_hypothesis;
    best_hypothesis.hypothesis_id = -1;
    best_hypothesis.dists_sum = std::numeric_limits<float>::max();
    best_hypothesis.n_inliers = 0;
    for (auto hypothesis_id : kv.second) {
      const RansacHypothesis& hypothesis = id_to_hypothesis[hypothesis_id];
      if ((hypothesis.n_inliers >= n_min_inliers) &&
          ((hypothesis.n_inliers > best_hypothesis.n_inliers) ||
           (hypothesis.n_inliers == best_hypothesis.n_inliers &&
            hypothesis.dists_sum < best_hypothesis.dists_sum))) {
        best_hypothesis = hypothesis;
      }
    }
    if (best_hypothesis.hypothesis_id > 0) {
      best_hypotheses.push_back(best_hypothesis.hypothesis_id);
      for (auto match : best_hypothesis.matches_inliers_uniqs) {
        inlier_matches_cand1.push_back(match.c1);
        inlier_matches_cand2.push_back(match.c2);
      }
    }
  }
  // py::dict outputs;
  // outputs["inlier_matches_cand1"] =
  // py::array_t<int>(py::cast(inlier_matches_cand1));
  // outputs["inlier_matches_cand2"] =
  // py::array_t<int>(py::cast(inlier_matches_cand2)); outputs["best_hypotheses"]
  // = py::array_t<int>(py::cast(best_hypotheses));
  candidates_outputs.inlier_c1 = inlier_matches_cand1;
  candidates_outputs.inlier_c2 = inlier_matches_cand2;
  candidates_outputs.best_hypo = best_hypotheses;
  // return outputs;
}

static void scatter_argmin(const std::vector<float> array_,
                           const std::vector<int> expand_ids_,
                           std::vector<int>& vector_best_ids) {
  // auto array_ = array.unchecked<1>();
  // auto expand_ids_ = expand_ids.unchecked<1>();

  std::unordered_map<int, float> lowest_values;
  std::unordered_map<int, int> best_ids;
  for (int n = 0; n < array_.size(); n++) {
    const int& expand_id = expand_ids_[n];
    const float& value = array_[n];
    const auto& exists = best_ids.find(expand_id);
    if (exists == best_ids.end()) {
      best_ids[expand_id] = n;
      lowest_values[expand_id] = value;
    }
    if (value < lowest_values[expand_id]) {
      best_ids[expand_id] = n;
      lowest_values[expand_id] = value;
    }
  }

  // std::vector<int> vector_best_ids;
  for (auto n = 0; n < best_ids.size(); n++) {
    vector_best_ids.push_back(best_ids[n]);
  }
  // auto array_best_ids = py::array_t<int>(py::cast(vector_best_ids));
  // return array_best_ids;
}

static void expand_ids_for_symmetry(const std::vector<int>& labels,
                                    std::unordered_map<int, int> n_symmetries,
                                    Symmetry_info* sym_info) {
  std::vector<int> ids_expand, labels_expand, sym_ids;
  for (auto n = 0; n < labels.size(); n++) {
    for (int k = 0; k < n_symmetries[labels[n]]; k++) {
      ids_expand.push_back(n);
      labels_expand.push_back(labels[n]);
      sym_ids.push_back(k);
    }
  }
  sym_info->labels_expand = labels_expand;
  sym_info->ids_expand = ids_expand;
  sym_info->sym_ids = sym_ids;
}

static torch::Tensor invert_T(torch::Tensor T) {
  torch::Tensor R, t, R_inv, t_inv, T_inv;
  R = T.index({"...", Slice(None, 3), Slice(None, 3)});
  t = T.index({"...", Slice(None, 3), Slice(3, 4)});
  R_inv = R.transpose(-2, -1);
  t_inv = -torch::matmul(R_inv, t);
  T_inv = T.clone();
  T_inv.index({"...", Slice(None, 3), Slice(None, 3)}) = R_inv;
  T_inv.index({"...", Slice(None, 3), Slice(3, 4)}) = t_inv;

  return T_inv;
}

static torch::Tensor transform_pts(torch::Tensor T, torch::Tensor pts) {
  // T: (bsz, 4, 4)
  // pts: (bsz, n_pts, 3)
  int bsz = T.size(0);
  int n_pts = pts.size(1);
  assert(pts.size(0) == bsz && pts.size(1) == n_pts && pts.size(2) == 3);
  if (T.dim() == 4) {
    pts = pts.unsqueeze(1);
    assert(T.size(1) == 4 && T.size(2) == 4);
  } else if (T.dim() == 3) {
    assert(T.size(1) == 4 && T.size(2) == 4);
  } else
    std::cerr << "Unsupported shape for T" << T.sizes() << std::endl;
  pts = pts.unsqueeze(-1);  //(bsz, n_pts, 3, 1)
  T = T.unsqueeze(-3);      //(bsz, 1, 4, 4)
  // torch::Tensor T1 = T.index({"...", Slice(None, 3), Slice(None, 3)});
  // std::cout << T1.sizes() << std::endl;
  // std::cout << pts.sizes() << std::endl;
  torch::Tensor pts_transformed =
      torch::matmul(T.index({"...", Slice(None, 3), Slice(None, 3)}), pts) +
      T.index({"...", Slice(None, 3), Slice(3, 4)});
  return pts_transformed.squeeze(-1);
}

static torch::Tensor symmetric_distance_batched_fast(torch::Tensor T1,
                                                     torch::Tensor T2,
                                                     std::vector<int> labels,
                                                     Mesh_database mesh_db) {
  int bsz = T1.size(0);
  assert(T1.size(1) == 4 && T1.size(2) == 4);
  assert(T2.size(0) == bsz && T2.size(1) == 4 && T2.size(2) == 4);
  assert(labels.size() == bsz);

  if (bsz == 0) return torch::empty(0);

  // meshes = mesh_db.select(labels);
  // points = meshes.points;
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor labels_tensor =
      torch::from_blob(labels.data(), {static_cast<int>(labels.size())}, opts)
          .clone()
          .to(torch::kInt64);
  torch::Tensor symmetry_tensor =
      torch::index(mesh_db.sym_tensor, {labels_tensor});
  torch::Tensor point_tensor = torch::index(mesh_db.kpts_3d, {labels_tensor});
  torch::Tensor T1_points = transform_pts(
      torch::matmul(T1.unsqueeze(1), symmetry_tensor), point_tensor);
  torch::Tensor T2_points = transform_pts(T2, point_tensor).unsqueeze(1);
  torch::Tensor dists_squared = torch::square(T1_points - T2_points).sum(-1);
  // std::cout << "dist squared" << dists_squared.mean(-1).index({12}) <<
  // std::endl;
  torch::Tensor best_sym_id = dists_squared.mean(-1).argmin(1);
  // std::cout << "best sym id" << best_sym_id.index({12}) << std::endl;
  torch::Tensor min_dists =
      torch::sqrt(
          torch::index(dists_squared, {torch::arange(bsz), best_sym_id}))
          .mean(-1);
  // S12 = meshes.symmetries[torch.arange(bsz), best_sym_id];
  return min_dists;
}

static void estimate_camera_poses(torch::Tensor TC1Oa, torch::Tensor TC2Ob,
                                  std::vector<int> labels_ab,
                                  torch::Tensor TC1Og, torch::Tensor TC2Od,
                                  std::vector<int> labels_gd,
                                  Mesh_database mesh_db, torch::Tensor& TC1C2) {
  // Assume(TC1Oa and TC2Ob), (TC1Og, TC2Od)are the same.
  // Notation differ from the paper, paper(code)
  // we have 1(a), 2(b), a(alpha), b(beta), g(gamma), d(delta)
  int bsz = TC1Oa.size(0);
  assert(TC1Oa.size(1) == 4 && TC1Oa.size(2) == 4);
  assert(TC2Ob.size(0) == bsz && TC2Ob.size(1) == 4 && TC2Ob.size(2) == 4);
  assert(TC1Og.size(0) == bsz && TC1Og.size(1) == 4 && TC1Og.size(2) == 4);
  assert(TC2Od.size(0) == bsz && TC2Od.size(1) == 4 && TC2Od.size(2) == 4);
  assert(labels_ab.size() == bsz);
  assert(labels_gd.size() == bsz);

  torch::Tensor TObC2 = invert_T(TC2Ob);

  // meshes_ab = mesh_db.select(labels_ab);
  Symmetry_info sym_info;
  expand_ids_for_symmetry(labels_ab, mesh_db.n_sym_mapping, &sym_info);
  std::vector<int> labels_gd_expand;
  for (auto expand_id : sym_info.ids_expand)
    labels_gd_expand.push_back(labels_gd[expand_id]);
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor ids_expand =
      torch::from_blob(sym_info.ids_expand.data(),
                       {static_cast<int>(sym_info.ids_expand.size())}, opts)
          .clone()
          .to(torch::kInt64);
  torch::Tensor labels_expandtensor =
      torch::from_blob(sym_info.labels_expand.data(),
                       {static_cast<int>(sym_info.labels_expand.size())}, opts)
          .clone()
          .to(torch::kInt64);
  torch::Tensor symids_tensor =
      torch::from_blob(sym_info.sym_ids.data(),
                       {static_cast<int>(sym_info.sym_ids.size())}, opts)
          .clone()
          .to(torch::kInt64);
  // std::cout << "tensor preparation done" << std::endl;
  // std::cout << labels_expandtensor.sizes() << std::endl;
  // std::cout << symids_tensor.sizes() << std::endl;
  torch::Tensor sym_expand =
      torch::index(mesh_db.sym_tensor, {labels_expandtensor, symids_tensor});
  // std::cout << "sym expand done" << std::endl;
  torch::Tensor dists = symmetric_distance_batched_fast(
      torch::index(TC1Og, {ids_expand}),
      torch::bmm(
          torch::bmm(torch::bmm(torch::index(TC1Oa, {ids_expand}), sym_expand),
                     torch::index(TObC2, {ids_expand})),
          torch::index(TC2Od, {ids_expand})),
      labels_gd_expand, mesh_db);
  // std::cout << "distance calculation done" << std::endl;
  // Create a vector to store the tensor data
  // Convert the tensor to a contiguous array
  dists = dists.contiguous();
  // std::cout << dists << std::endl;
  // Access the underlying data pointer
  auto data_ptr = dists.data_ptr<float>();

  // Create the std::vector and copy the data
  std::vector<float> dis_vec(dists.numel());
  std::copy(data_ptr, data_ptr + dists.numel(), dis_vec.begin());
  std::vector<int> min_ids;
  scatter_argmin(dis_vec, sym_info.ids_expand, min_ids);
  // std::cout << sym_info.ids_expand.size() << std::endl;
  // std::cout << "scatter argmin done" << std::endl;
  std::vector<int> min_symids;
  for (int idx_minid = 0; idx_minid < min_ids.size(); idx_minid++)
    min_symids.push_back(sym_info.sym_ids[min_ids[idx_minid]]);
  torch::Tensor labels_ab_tensor =
      torch::from_blob(labels_ab.data(), {static_cast<int>(labels_ab.size())},
                       opts)
          .clone()
          .to(torch::kInt64);
  torch::Tensor min_symid_tensor =
      torch::from_blob(min_symids.data(), {static_cast<int>(min_symids.size())},
                       opts)
          .clone()
          .to(torch::kInt64);
  // std::cout << "sym ids done" << std::endl;
  // std::cout << mesh_db.sym_tensor.sizes() << std::endl;
  // std::cout << labels_ab_tensor.sizes() << std::endl;
  // std::cout << min_symid_tensor.sizes() << std::endl;
  torch::Tensor S_Oa_star =
      torch::index(mesh_db.sym_tensor, {labels_ab_tensor, min_symid_tensor});

  // std::cout << "SOa star ids done" << std::endl;
  TC1C2 = torch::bmm(torch::bmm(TC1Oa, S_Oa_star), TObC2);
}

static void estimate_camera_poses_batch(Ransac_candidates candidates,
                                        Mesh_database mesh_db,
                                        torch::Tensor& all_TC1C2,
                                        int bsz = 1024) {
  int n_tot = candidates.match1_cand1.size();
  // std::cout << "total number: " + std::to_string(n_tot) << std::endl;
  int n_batch = std::max(1, int(n_tot / bsz));
  // std::cout << "batch number: " + std::to_string(n_batch) << std::endl;
  int batch_size;
  // ids_split = np.array_split(np.arange(n_tot), n_batch);
  // all_TC1C2 = [];
  torch::Tensor TC1Oa, TC2Ob, TC1Og, TC2Od, TC1C2, matrix_tensor;
  std::vector<int> labels_ab, labels_gd;
  all_TC1C2 = torch::zeros({n_tot, 4, 4});
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  for (int batch_idx = 0; batch_idx < n_batch; batch_idx++) {
    std::vector<int> ids_ab(candidates.match1_cand1.begin() + batch_idx * bsz,
                            candidates.match1_cand1.begin() +
                                std::min(batch_idx * bsz + bsz, n_tot));
    // std::cout << "ids_ab: " << ids_ab << std::endl;
    std::vector<int> ids_gd(candidates.match2_cand1.begin() + batch_idx * bsz,
                            candidates.match2_cand1.begin() +
                                std::min(batch_idx * bsz + bsz, n_tot));
    // std::cout << "ids_gd: " << ids_gd << std::endl;
    batch_size = std::min(bsz, n_tot - batch_idx * bsz);

    TC1Oa = torch::zeros({batch_size, 4, 4});
    TC2Ob = torch::zeros({batch_size, 4, 4});
    TC1Og = torch::zeros({batch_size, 4, 4});
    TC2Od = torch::zeros({batch_size, 4, 4});
    TC1C2 = torch::zeros({batch_size, 4, 4});
    labels_ab.clear();
    labels_gd.clear();
    for (int detect_idx = 0; detect_idx < batch_size; detect_idx++) {
      labels_ab.push_back(candidates.labels[ids_ab[detect_idx]]);
      labels_gd.push_back(candidates.labels[ids_gd[detect_idx]]);
      TC1Oa.index({detect_idx}) = candidates.poses.index(
          {candidates.match1_cand1[batch_idx * bsz + detect_idx]});
      // TC1Oa = detection_poses[seeds['match1_cand1'][ids]];
      TC2Ob.index({detect_idx}) = candidates.poses.index(
          {candidates.match1_cand2[batch_idx * bsz + detect_idx]});
      // TC2Ob = detection_poses[seeds['match1_cand2'][ids]];
      TC1Og.index({detect_idx}) = candidates.poses.index(
          {candidates.match2_cand1[batch_idx * bsz + detect_idx]});
      // TC1Og = detection_poses[seeds['match2_cand1'][ids]];
      TC2Od.index({detect_idx}) = candidates.poses.index(
          {candidates.match2_cand2[batch_idx * bsz + detect_idx]});
      // TC2Od = detection_poses[seeds['match2_cand2'][ids]];
    }
    // std::cout << "TC1Oa" << TC1Oa << std::endl;
    // std::cout << "TC2Ob" << TC2Ob << std::endl;
    // std::cout << "TC1Og" << TC1Og << std::endl;
    // std::cout << "TC2Od" << TC2Od << std::endl;
    // std::cout << "preparation done" << std::endl;
    // std::cout << "labels_ab: " << labels_ab << std::endl;
    // std::cout << "labels_gd: " << labels_gd << std::endl;
    estimate_camera_poses(TC1Oa, TC2Ob, labels_ab, TC1Og, TC2Od, labels_gd,
                          mesh_db, TC1C2);
    // std::cout << "estimation done" << std::endl;
    // all_TC1C2.append(TC1C2);
    all_TC1C2.index({Slice(batch_idx * bsz, batch_idx * bsz + batch_size)}) =
        TC1C2;
  }
  // return torch.cat(all_TC1C2, dim=0);
}

static torch::Tensor score_tmatches(torch::Tensor TC1Oa, torch::Tensor TC2Ob,
                                    torch::Tensor TC1C2,
                                    std::vector<int> labels_ab,
                                    Mesh_database mesh_db) {
  torch::Tensor TWOa = TC1Oa;
  torch::Tensor TWOb = torch::bmm(TC1C2, TC2Ob);

  torch::Tensor dists =
      symmetric_distance_batched_fast(TWOa, TWOb, labels_ab, mesh_db);
  return dists;
}

static torch::Tensor score_tmaches_batch(Ransac_candidates candidates,
                                         torch::Tensor TC1C2,
                                         Mesh_database mesh_db,
                                         int bsz = 4096) {
  int n_tot = candidates.cand1.size();
  // std::cout << "total number: " + std::to_string(n_tot) << std::endl;
  int n_batch = std::max(1, int(n_tot / bsz));
  // std::cout << "batch number: " + std::to_string(n_batch) << std::endl;
  // ids_split = np.array_split(np.arange(n_tot), n_batch)
  // all_dists = []
  torch::Tensor TC1Oa, TC2Ob, TC1C2_, dists, all_dists;
  all_dists = torch::zeros({n_tot});
  int batch_size;
  for (int batch_idx = 0; batch_idx < n_batch; batch_idx++) {
    std::vector<int> ids(
        candidates.cand1.begin() + batch_idx * bsz,
        candidates.cand1.begin() + std::min(batch_idx * bsz + bsz, n_tot));
    std::vector<int> labels;
    batch_size = std::min(bsz, n_tot - batch_idx * bsz);
    TC1Oa = torch::zeros({batch_size, 4, 4});
    TC2Ob = torch::zeros({batch_size, 4, 4});
    TC1C2_ = torch::zeros({batch_size, 4, 4});
    for (int idx = 0; idx < ids.size(); idx++) {
      labels.push_back(candidates.labels[ids[idx]]);
      // labels =
      // candidates.infos['label'].iloc[tmatches['cand1'][ids]].values;
      TC1Oa.index({idx}) =
          candidates.poses.index({candidates.cand1[batch_idx * bsz + idx]});
      TC2Ob.index({idx}) =
          candidates.poses.index({candidates.cand2[batch_idx * bsz + idx]});
      TC1C2_.index({idx}) =
          TC1C2.index({candidates.hypothesis_id[batch_idx * bsz + idx]});
    }
    // std::cout << "score tmatches" << std::endl;
    dists = score_tmatches(TC1Oa, TC2Ob, TC1C2_, labels, mesh_db);

    all_dists.index({Slice(batch_idx * bsz, batch_idx * bsz + batch_size)}) =
        dists;
  }
  return all_dists;
}

static void get_best_viewpair_pose_est(torch::Tensor TC1C2,
                                       Ransac_candidates candidates,
                                       Ransac_inlier inliers,
                                       Results& pose_results) {
  std::vector<int> best_hypotheses = inliers.best_hypo;
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor best_ids =
      torch::from_blob(best_hypotheses.data(),
                       {static_cast<int>(best_hypotheses.size())}, opts)
          .clone()
          .to(torch::kInt64);
  torch::Tensor TC1C2_best = TC1C2.index({best_ids});
  std::vector<int> best_view1, best_view2;
  // std::cout << best_hypotheses.size() << std::endl;
  for (int best_id = 0; best_id < best_hypotheses.size(); best_id++) {
    best_view1.push_back(candidates.view1[best_hypotheses[best_id]]);
    best_view2.push_back(candidates.view2[best_hypotheses[best_id]]);
  }
  pose_results.view1 = best_view1;
  pose_results.view2 = best_view2;
  pose_results.TC1C2 = TC1C2_best;
}

/*
  static void scene_level_matching(candidates, inliers){
    cand1 = inliers['inlier_matches_cand1']
    cand2 = inliers['inlier_matches_cand2']
    edges = np.ones((len(cand1)), dtype=np.int)
    n_cand = len(candidates)
    graph = csr_matrix((edges, (cand1, cand2)), shape=(n_cand, n_cand))
    n_components, ids = connected_components(graph, directed=True, connection='strong')

    component_size = defaultdict(lambda: 0)
    for idx in ids:
        component_size[idx] += 1
    obj_n_cand = np.empty(len(ids), dtype=np.int)
    for n, idx in enumerate(ids):
        obj_n_cand[n] = component_size[idx]

    cand_infos = candidates.infos.copy()
    cand_infos['component_id'] = ids
    keep_cand = obj_n_cand >= 2
    cand_infos = cand_infos[keep_cand].reset_index(drop=True)
    for n, (comp_id, group) in enumerate(cand_infos.groupby('component_id')):
        cand_infos.loc[group.index, 'component_id'] = n
    cand_infos = cand_infos.rename(columns={'component_id': 'obj_id'})

    matched_candidates = tc.PandasTensorCollection(infos=cand_infos,
                                                   poses=candidates.poses[cand_infos['cand_id'].values])
  }
  */

  static void multiview_candidate_matching(Ransac_candidates candidates,
                                    Mesh_database mesh_db,
                                    Results& pose_results, std::vector<int> &inlier_ids, int model_bsz = 1e3,
                                   int score_bsz = 1e5, float dist_threshold = 0.02, int n_ransac_iter = 2000,
                                   int n_min_inliers = 3) {

        // Estimating camera poses using RANSAC
        //std::cout << "make ransac info" << std::endl;
        make_ransac_infos(candidates, n_ransac_iter, 0);
        torch::Tensor TC1C2;
        //std::cout << "estimate camera poses" << std::endl;
        estimate_camera_poses_batch(candidates, mesh_db, TC1C2, model_bsz);
        //std::cout << "match scores" << std::endl;
        //std::cout << TC1C2 << std::endl;
        torch::Tensor dists = score_tmaches_batch(candidates, TC1C2, mesh_db,
                                score_bsz);
        dists = dists.contiguous();
        //std::cout << dists << std::endl;
        // Access the underlying data pointer
        auto data_ptr = dists.data_ptr<float>();

        // Create the std::vector and copy the data
        std::vector<float> dis_vec(dists.numel());
        std::copy(data_ptr, data_ptr + dists.numel(), dis_vec.begin());
        //std::cout << dists << std::endl;
        Ransac_inlier candidates_inlier;
        //std::cout << "find ransac inlier" << std::endl;
        //std::cout << "hypo num: " << candidates.hypothesis_id.size()<< std::endl;
        //std::cout << "dist num: " << dis_vec.size() << std::endl;
        //std::cout << "view1 "  << candidates.view1.size() << std::endl;
        //std::cout << "view2 " << candidates.view2.size() << std::endl;
        find_ransac_inliers(candidates.view1, candidates.view2,
                            candidates.hypothesis_id,
                            candidates.cand1,
                            candidates.cand2, dis_vec, dist_threshold,
                            n_min_inliers, candidates_inlier);
        //std::cout << "get best view pair" << std::endl;
        //std::cout << candidates_inlier.best_hypo.size() << std::endl;
        get_best_viewpair_pose_est(TC1C2, candidates, candidates_inlier,
                                   pose_results);

        // MERGE
        std::merge(candidates_inlier.inlier_c1.begin(),
                   candidates_inlier.inlier_c1.end(),
                   candidates_inlier.inlier_c2.begin(),
                   candidates_inlier.inlier_c2.end(),
                   std::back_inserter(inlier_ids));
        // Sort the vector
        std::sort(inlier_ids.begin(), inlier_ids.end());

        // Move all duplicates to last of vector
        auto it = std::unique(inlier_ids.begin(), inlier_ids.end());

        // Remove all duplicates
        inlier_ids.erase(it, inlier_ids.end());
        //std::cout << "candidates" << std::endl;
        //std::cout << candidates.view_ids << std::endl;
        //std::cout << "inliers" << std::endl;
        //std::cout << inlier_ids << std::endl;
        
        //std::cout << candidates_inlier.inlier_c1 << std::endl;
        //std::cout << candidates_inlier.inlier_c2 << std::endl;
        //std::cout << "view 1 " << pose_results.view1 << std::endl;
        //std::cout << "view 2 " << pose_results.view2 << std::endl;
        //filtered_candidates = scene_level_matching(candidates, inliers);
        //scene_infos = make_obj_infos(filtered_candidates)
  }

  } 

// namespace cosypose

#endif  // _COSYPOSE_H_

