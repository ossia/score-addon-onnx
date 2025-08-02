#pragma once
#include <QDebug>
#include <QImage>
#include <QPainter>

#include <cmath>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace Onnx
{

struct Peak
{
  float x, y; // Use float for sub-pixel precision
  float confidence;
  int id;
};

struct Keypoint
{
  float x, y;
  float confidence;
};

struct Person
{
  std::array<Keypoint, 18> keypoints;
  float total_score;
};

struct TrtTopology
{
  static const std::vector<std::array<int, 4>>& getCOCOTopology()
  {
    // Format: [paf_channel_x, paf_channel_y, source_joint, sink_joint]
    static const std::vector<std::array<int, 4>> topology = {
        {0, 1, 17, 0},    // neck -> nose
        {2, 3, 0, 1},     // nose -> left_eye
        {4, 5, 0, 2},     // nose -> right_eye
        {6, 7, 1, 3},     // left_eye -> left_ear
        {8, 9, 2, 4},     // right_eye -> right_ear
        {10, 11, 17, 5},  // neck -> left_shoulder
        {12, 13, 17, 6},  // neck -> right_shoulder
        {14, 15, 5, 7},   // left_shoulder -> left_elbow
        {16, 17, 7, 9},   // left_elbow -> left_wrist
        {18, 19, 6, 8},   // right_shoulder -> right_elbow
        {20, 21, 8, 10},  // right_elbow -> right_wrist
        {22, 23, 17, 11}, // neck -> left_hip
        {24, 25, 17, 12}, // neck -> right_hip
        {26, 27, 11, 13}, // left_hip -> left_knee
        {28, 29, 13, 15}, // left_knee -> left_ankle
        {30, 31, 12, 14}, // right_hip -> right_knee
        {32, 33, 14, 16}, // right_knee -> right_ankle
        {34, 35, 5, 6},   // left_shoulder -> right_shoulder
        {36, 37, 11, 12}, // left_hip -> right_hip
        {38, 39, 5, 11},  // left_shoulder -> left_hip
        {40, 41, 6, 12}   // right_shoulder -> right_hip
    };
    return topology;
  }

  static const std::vector<std::string>& getKeypointNames()
  {
    static const std::vector<std::string> names
        = {"nose",
           "left_eye",
           "right_eye",
           "left_ear",
           "right_ear",
           "left_shoulder",
           "right_shoulder",
           "left_elbow",
           "right_elbow",
           "left_wrist",
           "right_wrist",
           "left_hip",
           "right_hip",
           "left_knee",
           "right_knee",
           "left_ankle",
           "right_ankle",
           "neck"};
    return names;
  }
};

class TRT_pose
{
public:
  struct Config
  {
    float confidence_threshold = 0.1f;
    float paf_threshold = 0.1f;
    int max_peaks_per_part = 100;
    int peak_window_size = 3;
    int paf_integral_samples = 7;
  };

  TRT_pose(const Config& config)
      : config_(config)
  {
  }

  std::vector<Person> processOutput(
      const float* cmap_data, // [1, 18, 128, 128] - confidence maps
      const float* paf_data,  // [1, 42, 128, 128] - part affinity fields
      int height = 128,
      int width = 128);

  QImage visualizePoses(
      const std::vector<Person>& persons,
      int img_width,
      int img_height,
      int model_width = 128,
      int model_height = 128);

  QImage visualizeConfidenceMaps(
      const float* cmap_data,
      int cmap_height,
      int cmap_width,
      int img_width,
      int img_height,
      float max_confidence,
      const std::vector<int>& parts_to_show = {} // Empty = show all parts
  );

  QImage visualizePAF(
      const float* paf_data,
      int paf_height,
      int paf_width,
      int img_width,
      int img_height,
      float scale_factor = 8.0f, // Skip pixels for arrow spacing
      float magnitude_threshold = 0.1f);

  QImage visualizePeaks(
      const std::vector<std::vector<Peak>>& peaks,
      int img_width,
      int img_height,
      int model_width = 128,
      int model_height = 128);

  QImage visualizePAFScores(
      const std::vector<std::vector<Peak>>& peaks,
      const std::vector<std::vector<std::vector<float>>>& paf_scores,
      int img_width,
      int img_height,
      int model_width = 128,
      int model_height = 128,
      float score_threshold = 0.1f);

  // Peak detection (public for debugging)
  std::vector<std::vector<Peak>>
  findPeaks(const float* cmap_data, int height, int width);

  // Peak refinement (public for debugging)
  void refinePeaks(
      std::vector<std::vector<Peak>>& peaks,
      const float* cmap_data,
      int height,
      int width);

  // PAF scoring (public for debugging)
  std::vector<std::vector<std::vector<float>>> scorePAF(
      const std::vector<std::vector<Peak>>& peaks,
      const float* paf_data,
      int height,
      int width);

  // Simple PAF scoring for debugging
  std::vector<std::vector<std::vector<float>>> scorePAFSimple(
      const std::vector<std::vector<Peak>>& peaks,
      const float* paf_data,
      int height,
      int width);

  Config config_;

  // Hungarian assignment
  std::vector<std::vector<std::array<int, 2>>> assignConnections(
      const std::vector<std::vector<std::vector<float>>>& paf_scores,
      const std::vector<std::vector<Peak>>& peaks);

  std::vector<std::vector<std::array<int, 2>>> assignConnectionsGreedy(
      const std::vector<std::vector<std::vector<float>>>& paf_scores,
      const std::vector<std::vector<Peak>>& peaks);

  // Connect parts to form persons
  std::vector<Person> connectParts(
      const std::vector<std::vector<Peak>>& peaks,
      const std::vector<std::vector<std::array<int, 2>>>& connections);

  // Utility functions
  bool isLocalMaximum(
      const float* data,
      int h,
      int w,
      int height,
      int width,
      float threshold);
  float bilinearInterpolate(
      const float* data,
      float x,
      float y,
      int height,
      int width);
  float evaluateLineIntegral(
      const float* paf_x,
      const float* paf_y,
      float x1,
      float y1,
      float x2,
      float y2,
      int height,
      int width);
  std::vector<int>
  hungarianAssignment(const std::vector<std::vector<float>>& cost_matrix);
  std::vector<Person>
  applyNMS(const std::vector<Person>& persons, float overlap_threshold);
  bool isSpatiallyCoherent(const Person& person);
  void debugPeakDetection(
      const float* cmap_data,
      int height,
      int width,
      int part_id);
};

// Implementation
inline std::vector<std::vector<Peak>>
TRT_pose::findPeaks(const float* cmap_data, int height, int width)
{
  std::vector<std::vector<Peak>> all_peaks(18);

  for (int part = 0; part < 18; part++)
  {
    const float* part_map = cmap_data + part * height * width;
    std::vector<Peak>& peaks = all_peaks[part];

    // Find max value for this part to debug
    float part_max = 0.0f;
    for (int i = 0; i < height * width; i++)
    {
      part_max = std::max(part_max, part_map[i]);
    }

    // Use more lenient criteria for debugging
    int found_peaks = 0;

    // First, find the absolute maximum to compare with detected peaks
    // First, find the absolute maximum to compare with detected peaks
    float absolute_max = 0.0f;
    int max_h = 0, max_w = 0;
    for (int h = 0; h < height; h++)
    {
      for (int w = 0; w < width; w++)
      {
        float val = part_map[h * width + w];
        if (val > absolute_max)
        {
          absolute_max = val;
          max_h = h;
          max_w = w;
        }
      }
    }

    // TEMPORARY: Just find the absolute maximum for debugging
    if (absolute_max > config_.confidence_threshold)
    {
      peaks.push_back(
          {static_cast<float>(max_w),
           static_cast<float>(max_h),
           absolute_max,
           0});
      // qDebug() << "Part" << part << "added absolute max peak at (" << max_w
      //          << "," << max_h << ") conf:" << absolute_max;
    }

// DISABLED BECAUSE IT DOES NOT WORK:
#if 0
    // Use the improved peak detection we developed earlier
    // Simple approach: collect all pixels above threshold, sort by confidence, then apply non-maximum suppression
    std::vector<Peak> candidates;
    
    for (int h = 0; h < height; h++)
    {
      for (int w = 0; w < width; w++)
      {
        float val = part_map[h * width + w];
        if (val > config_.confidence_threshold)
        {
          candidates.push_back({static_cast<float>(w), static_cast<float>(h), val, 0});
        }
      }
    }
    
    // Sort by confidence (highest first)
    std::sort(candidates.begin(), candidates.end(), [](const Peak& a, const Peak& b) {
      return a.confidence > b.confidence;
    });
    
    // Apply non-maximum suppression: keep peaks that are far enough apart
    float min_distance = static_cast<float>(config_.peak_window_size * 2); // Increase minimum distance
    
    for (const auto& candidate : candidates)
    {
      // Skip peaks too close to edges (likely artifacts)
      if (candidate.x < 5 || candidate.x > width - 5 || candidate.y < 5
          || candidate.y > height - 5)
      {
        continue;
      }

      bool too_close = false;
      for (const auto& existing_peak : peaks)
      {
        float dx = candidate.x - existing_peak.x;
        float dy = candidate.y - existing_peak.y;
        float distance = std::sqrt(dx * dx + dy * dy);
        
        if (distance < min_distance)
        {
          too_close = true;
          break;
        }
      }
      
      if (!too_close)
      {
        Peak new_peak = candidate;
        new_peak.id = static_cast<int>(peaks.size());
        peaks.push_back(new_peak);
        found_peaks++;
        
        // Limit number of peaks per part
        if (peaks.size() >= config_.max_peaks_per_part)
        {
          break;
        }
      }
    }

    // if (part == 0 || part == 17)
    // { // Debug nose and neck
    //   qDebug() << "Part" << part << "absolute max:" << absolute_max << "at ("
    //            << max_w << "," << max_h << ")";
    // }
#endif
    // Keep only top peaks
    if (peaks.size() > config_.max_peaks_per_part)
    {
      std::partial_sort(
          peaks.begin(),
          peaks.begin() + config_.max_peaks_per_part,
          peaks.end(),
          [](const Peak& a, const Peak& b)
          { return a.confidence > b.confidence; });
      peaks.resize(config_.max_peaks_per_part);
    }
#if 0
    // Debug all parts with peaks
    if (!peaks.empty())
    {
      qDebug() << "Part" << part << "("
               << TrtTopology::getKeypointNames()[part].c_str() << ") found"
               << peaks.size() << "peaks";
      for (size_t i = 0; i < std::min(size_t(3), peaks.size()); i++)
      {
        qDebug() << "  Peak" << i << "at (" << peaks[i].x << "," << peaks[i].y
                 << ") conf:" << peaks[i].confidence;

        // Flag suspicious peaks near edges
        if (peaks[i].x < 5 || peaks[i].x > width - 5 || peaks[i].y < 5
            || peaks[i].y > height - 5)
        {
          qDebug() << "    WARNING: Peak near edge of image!";
        }
      }
    }

    if (part == 0 || part == 17)
    { // Debug nose and neck
      // qDebug() << "Part" << part << "max:" << part_max
      //          << "peaks:" << peaks.size();

      // Show first few peaks for debugging
      for (size_t i = 0; i < std::min(size_t(3), peaks.size()); i++)
      {
        // qDebug() << "  Peak" << i << "at (" << peaks[i].x << "," << peaks[i].y
        //          << ") conf:" << peaks[i].confidence;
      }
    }
    // qDebug() << part << " => " << peaks.size() << peaks[0].x << peaks[0].y
    //          << peaks[0].confidence;
#endif
  }

  return all_peaks;
}

inline bool TRT_pose::isLocalMaximum(
    const float* data,
    int h,
    int w,
    int height,
    int width,
    float threshold)
{
  float center_val = data[h * width + w];
  if (center_val < threshold)
    return false;

  int window = config_.peak_window_size;
  for (int dh = -window; dh <= window; dh++)
  {
    for (int dw = -window; dw <= window; dw++)
    {
      if (dh == 0 && dw == 0)
        continue;

      int nh = h + dh;
      int nw = w + dw;
      if (nh >= 0 && nh < height && nw >= 0 && nw < width)
      {
        if (data[nh * width + nw] > center_val)
        {
          return false;
        }
      }
    }
  }
  return true;
}

inline void TRT_pose::refinePeaks(
    std::vector<std::vector<Peak>>& peaks,
    const float* cmap_data,
    int height,
    int width)
{
  for (int part = 0; part < 18; part++)
  {
    const float* part_map = cmap_data + part * height * width;

    for (auto& peak : peaks[part])
    {
      // Quadratic refinement
      int x = static_cast<int>(peak.x), y = static_cast<int>(peak.y);
      if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
      {
        float dx = (part_map[y * width + x + 1] - part_map[y * width + x - 1])
                   * 0.5f;
        float dy
            = (part_map[(y + 1) * width + x] - part_map[(y - 1) * width + x])
              * 0.5f;
        float dxx = part_map[y * width + x + 1] - 2 * part_map[y * width + x]
                    + part_map[y * width + x - 1];
        float dyy = part_map[(y + 1) * width + x] - 2 * part_map[y * width + x]
                    + part_map[(y - 1) * width + x];

        if (std::abs(dxx) > 1e-6 && std::abs(dyy) > 1e-6)
        {
          float offset_x = -dx / dxx;
          float offset_y = -dy / dyy;

          // Clamp offsets
          offset_x = std::max(-1.0f, std::min(1.0f, offset_x));
          offset_y = std::max(-1.0f, std::min(1.0f, offset_y));

          peak.x = std::max(
              0.0f, std::min(static_cast<float>(width - 1), x + offset_x));
          peak.y = std::max(
              0.0f, std::min(static_cast<float>(height - 1), y + offset_y));
        }
      }
    }
  }
}

inline float TRT_pose::evaluateLineIntegral(
    const float* paf_x,
    const float* paf_y,
    float x1,
    float y1,
    float x2,
    float y2,
    int height,
    int width)
{
  float score = 0.0f;
  int count = 0;

  // Debug: check if connection is very short
  float dx_total = x2 - x1;
  float dy_total = y2 - y1;
  float total_distance = std::sqrt(dx_total * dx_total + dy_total * dy_total);

  if (total_distance < 1.0f)
  {
    // For very short connections, just sample the PAF at the midpoint
    float mid_x = (x1 + x2) / 2.0f;
    float mid_y = (y1 + y2) / 2.0f;

    if (mid_x >= 0 && mid_x < width - 1 && mid_y >= 0 && mid_y < height - 1)
    {
      float vx = bilinearInterpolate(paf_x, mid_x, mid_y, height, width);
      float vy = bilinearInterpolate(paf_y, mid_x, mid_y, height, width);

      float norm = std::sqrt(dx_total * dx_total + dy_total * dy_total);
      if (norm > 1e-6)
      {
        score = (vx * dx_total + vy * dy_total) / norm;
      }
    }
    return score;
  }

  // Normal case: sample along the line
  for (int i = 0; i < config_.paf_integral_samples; i++)
  {
    float t = static_cast<float>(i) / (config_.paf_integral_samples - 1);
    float x = x1 + t * dx_total;
    float y = y1 + t * dy_total;

    if (x >= 0 && x < width - 1 && y >= 0 && y < height - 1)
    {
      float vx = bilinearInterpolate(paf_x, x, y, height, width);
      float vy = bilinearInterpolate(paf_y, x, y, height, width);

      // Unit direction vector
      float dx = dx_total / total_distance;
      float dy = dy_total / total_distance;

      // Dot product with PAF vector
      float dot = vx * dx + vy * dy;
      score += dot;
      count++;
    }
  }

  // Average score along the line
  float avg_score = count > 0 ? score / count : 0.0f;

  // Use absolute value since PAF might be flipped
  avg_score = std::abs(avg_score);

  // Scale up since PAF values are very small (as discovered in debugging)
  avg_score = avg_score * 100.0f;

#if 0
  // Debug extreme cases
  static int debug_count = 0;
  if (debug_count < 20 && avg_score > 0.5f)
  {
    debug_count++;
    qDebug() << "Line integral: (" << x1 << "," << y1 << ") -> (" << x2 << ","
             << y2 << ")"
             << "distance:" << total_distance << "samples:" << count
             << "score:" << avg_score;
  }
#endif
  return avg_score;
}

inline float TRT_pose::bilinearInterpolate(
    const float* data,
    float x,
    float y,
    int height,
    int width)
{
  int x0 = static_cast<int>(x);
  int y0 = static_cast<int>(y);
  int x1 = std::min(x0 + 1, width - 1);
  int y1 = std::min(y0 + 1, height - 1);

  float fx = x - x0;
  float fy = y - y0;

  float v00 = data[y0 * width + x0];
  float v01 = data[y0 * width + x1];
  float v10 = data[y1 * width + x0];
  float v11 = data[y1 * width + x1];

  return v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy
         + v11 * fx * fy;
}

inline std::vector<std::vector<std::vector<float>>> TRT_pose::scorePAF(
    const std::vector<std::vector<Peak>>& peaks,
    const float* paf_data,
    int height,
    int width)
{

  const auto& topology = TrtTopology::getCOCOTopology();
  std::vector<std::vector<std::vector<float>>> paf_scores(topology.size());

  // Debug: count total scoring operations
  int total_scorings = 0;
  int zero_scores = 0;
  int negative_scores = 0;
  int positive_scores = 0;

  for (size_t link_idx = 0; link_idx < topology.size(); link_idx++)
  {
    const auto& link = topology[link_idx];
    int paf_x_idx = link[0];
    int paf_y_idx = link[1];
    int part_a = link[2];
    int part_b = link[3];

    const float* paf_x = paf_data + paf_x_idx * height * width;
    const float* paf_y = paf_data + paf_y_idx * height * width;

    const auto& peaks_a = peaks[part_a];
    const auto& peaks_b = peaks[part_b];

    paf_scores[link_idx].resize(peaks_a.size());
#if 0
    // Debug first few links
    if (link_idx < 5 && !peaks_a.empty() && !peaks_b.empty())
    {
      qDebug() << "Link" << link_idx << ":" << part_a << "->" << part_b
               << "testing" << peaks_a.size() << "x" << peaks_b.size()
               << "connections";
    }
#endif
    for (size_t i = 0; i < peaks_a.size(); i++)
    {
      paf_scores[link_idx][i].resize(peaks_b.size());
      for (size_t j = 0; j < peaks_b.size(); j++)
      {
        float score = evaluateLineIntegral(
            paf_x,
            paf_y,
            peaks_a[i].x, // Already float, no need to cast
            peaks_a[i].y,
            peaks_b[j].x,
            peaks_b[j].y,
            height,
            width);

        paf_scores[link_idx][i][j] = score;
#if 0
        total_scorings++;

        // Track original sign for debugging
        if (score == 0.0f)
          zero_scores++;
        else if (score < 0.0f)
          negative_scores++;
        else
          positive_scores++;

        // Debug high scores for first few links
        if (link_idx < 5 && score > 0.1f)
        {
          qDebug() << "  Good score:" << score << "for connection"
                   << "(" << peaks_a[i].x << "," << peaks_a[i].y << ") -> "
                   << "(" << peaks_b[j].x << "," << peaks_b[j].y << ")";
        }
#endif
      }
    }
  }
#if 0
  qDebug() << "PAF Scoring summary: Total:" << total_scorings
           << "Positive:" << positive_scores << "Zero:" << zero_scores
           << "Negative:" << negative_scores;
#endif
  return paf_scores;
}

inline std::vector<std::vector<std::vector<float>>> TRT_pose::scorePAFSimple(
    const std::vector<std::vector<Peak>>& peaks,
    const float* paf_data,
    int height,
    int width)
{
  const auto& topology = TrtTopology::getCOCOTopology();
  std::vector<std::vector<std::vector<float>>> paf_scores(topology.size());

  for (size_t link_idx = 0; link_idx < topology.size(); link_idx++)
  {
    const auto& link = topology[link_idx];
    int paf_x_idx = link[0];
    int paf_y_idx = link[1];
    int part_a = link[2];
    int part_b = link[3];

    const float* paf_x = paf_data + paf_x_idx * height * width;
    const float* paf_y = paf_data + paf_y_idx * height * width;

    const auto& peaks_a = peaks[part_a];
    const auto& peaks_b = peaks[part_b];

    paf_scores[link_idx].resize(peaks_a.size());

    for (size_t i = 0; i < peaks_a.size(); i++)
    {
      paf_scores[link_idx][i].resize(peaks_b.size());
      for (size_t j = 0; j < peaks_b.size(); j++)
      {
        // Simple scoring: just check PAF at midpoint
        float mid_x = (peaks_a[i].x + peaks_b[j].x) / 2.0f;
        float mid_y = (peaks_a[i].y + peaks_b[j].y) / 2.0f;

        float score = 0.0f;

        // Make sure midpoint is in bounds
        if (mid_x >= 0 && mid_x < width - 1 && mid_y >= 0
            && mid_y < height - 1)
        {
          // Get PAF vector at midpoint
          float vx = bilinearInterpolate(paf_x, mid_x, mid_y, height, width);
          float vy = bilinearInterpolate(paf_y, mid_x, mid_y, height, width);

          // Expected direction (normalized)
          float dx = peaks_b[j].x - peaks_a[i].x;
          float dy = peaks_b[j].y - peaks_a[i].y;
          float norm = std::sqrt(dx * dx + dy * dy);

          if (norm > 1e-6)
          {
            dx /= norm;
            dy /= norm;

            // Score is dot product
            score = vx * dx + vy * dy;

#if 0
            // Also consider PAF magnitude
            float paf_magnitude = std::sqrt(vx * vx + vy * vy);
            // Debug: analyze PAF values for first few connections
            if (link_idx < 3 && i == 0 && j == 0)
            {
              qDebug() << "PAF Debug - Link" << link_idx << ":"
                       << "PAF vec (" << vx << "," << vy << ")"
                       << "magnitude:" << paf_magnitude << "expected dir ("
                       << dx << "," << dy << ")"
                       << "dot product:" << score << "distance:" << norm;

              // Also check PAF values around the midpoint
              for (int dy = -1; dy <= 1; dy++)
              {
                for (int dx = -1; dx <= 1; dx++)
                {
                  float test_x = mid_x + dx;
                  float test_y = mid_y + dy;
                  if (test_x >= 0 && test_x < width && test_y >= 0
                      && test_y < height)
                  {
                    int idx = static_cast<int>(test_y) * width
                              + static_cast<int>(test_x);
                    float test_vx = paf_x[idx];
                    float test_vy = paf_y[idx];
                    qDebug() << "  PAF at (" << test_x << "," << test_y
                             << ") = (" << test_vx << "," << test_vy << ")";
                  }
                }
              }
            }
#endif
            // Try different scoring approach - use absolute value if PAF might be flipped
            float abs_score = std::abs(score);

            // Use the better of forward or reverse score
            score = abs_score;

            // Scale up the score since PAF values seem very small
            score = score * 100.0f;
          }
        }

        paf_scores[link_idx][i][j] = score;

        // Debug logging - lower threshold for debugging
        if (link_idx < 5 && score > 0.001f)
        {
          qDebug() << "Simple PAF score:" << score << "for link" << link_idx
                   << "(" << peaks_a[i].x << "," << peaks_a[i].y << ") -> "
                   << "(" << peaks_b[j].x << "," << peaks_b[j].y << ")";
        }
      }
    }
  }

  return paf_scores;
}

inline std::vector<int> TRT_pose::hungarianAssignment(
    const std::vector<std::vector<float>>& cost_matrix)
{
  if (cost_matrix.empty() || cost_matrix[0].empty())
  {
    return std::vector<int>(cost_matrix.size(), -1);
  }

  size_t rows = cost_matrix.size();
  size_t cols = cost_matrix[0].size();

  // Convert to maximization problem (Hungarian solves minimization)
  // Find max value to convert scores to costs
  float max_val = 0;
  for (const auto& row : cost_matrix)
  {
    for (float val : row)
    {
      max_val = std::max(max_val, val);
    }
  }

  // Create cost matrix (max_val - score) and pad to square
  size_t n = std::max(rows, cols);
  std::vector<std::vector<float>> cost(n, std::vector<float>(n, max_val + 1));

  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = 0; j < cols; j++)
    {
      cost[i][j] = max_val - cost_matrix[i][j];
    }
  }

  // Hungarian algorithm implementation
  std::vector<float> u(n + 1, 0), v(n + 1, 0); // Potentials
  std::vector<int> p(n + 1, 0), way(n + 1, 0); // Assignment and path

  for (size_t i = 1; i <= n; ++i)
  {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(n + 1, std::numeric_limits<float>::max());
    std::vector<bool> used(n + 1, false);

    do
    {
      used[j0] = true;
      int i0 = p[j0];
      float delta = std::numeric_limits<float>::max();
      int j1 = 0;

      for (size_t j = 1; j <= n; ++j)
      {
        if (!used[j])
        {
          float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j])
          {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta)
          {
            delta = minv[j];
            j1 = j;
          }
        }
      }

      for (size_t j = 0; j <= n; ++j)
      {
        if (used[j])
        {
          u[p[j]] += delta;
          v[j] -= delta;
        }
        else
        {
          minv[j] -= delta;
        }
      }

      j0 = j1;
    } while (p[j0] != 0);

    do
    {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  // Extract assignment and filter by threshold
  std::vector<int> assignment(rows, -1);
  for (size_t j = 1; j <= cols && j <= n; ++j)
  {
    if (p[j] <= static_cast<int>(rows))
    {
      int i = p[j] - 1;
      int col = j - 1;
      // Only assign if original score meets threshold
      if (cost_matrix[i][col] >= config_.paf_threshold)
      {
        assignment[i] = col;
      }
    }
  }

  return assignment;
}

inline std::vector<std::vector<std::array<int, 2>>>
TRT_pose::assignConnections(
    const std::vector<std::vector<std::vector<float>>>& paf_scores,
    const std::vector<std::vector<Peak>>& peaks)
{

  // Use greedy assignment instead of Hungarian for better stability
  return assignConnectionsGreedy(paf_scores, peaks);
}

inline std::vector<std::vector<std::array<int, 2>>>
TRT_pose::assignConnectionsGreedy(
    const std::vector<std::vector<std::vector<float>>>& paf_scores,
    const std::vector<std::vector<Peak>>& peaks)
{

  std::vector<std::vector<std::array<int, 2>>> connections(paf_scores.size());

  // Debug: count total assignments
  int total_candidates = 0;
  int assigned_connections = 0;

  for (size_t link_idx = 0; link_idx < paf_scores.size(); link_idx++)
  {
    const auto& scores = paf_scores[link_idx];
    if (scores.empty())
      continue;

    // Create list of all possible connections with their scores
    std::vector<std::tuple<float, int, int>>
        candidates; // score, peak_a_idx, peak_b_idx

    for (size_t i = 0; i < scores.size(); i++)
    {
      for (size_t j = 0; j < scores[i].size(); j++)
      {
        if (scores[i][j] > config_.paf_threshold)
        {
          candidates.emplace_back(
              scores[i][j], static_cast<int>(i), static_cast<int>(j));
          total_candidates++;
        }
      }
    }
#if 0
    // Debug first few links
    if (link_idx < 5 && !candidates.empty())
    {
      qDebug() << "AssignGreedy - Link" << link_idx << "has"
               << candidates.size() << "candidates above threshold"
               << config_.paf_threshold;
      // Show top candidates
      for (size_t k = 0; k < std::min(size_t(3), candidates.size()); k++)
      {
        qDebug() << "  Candidate" << k
                 << ": score=" << std::get<0>(candidates[k]) << "peaks"
                 << std::get<1>(candidates[k]) << "->"
                 << std::get<2>(candidates[k]);
      }
    }
#endif

    // Sort by score (highest first)
    std::sort(candidates.begin(), candidates.end(), std::greater<>());

    // Greedily select connections, ensuring each peak is used at most once
    std::vector<bool> used_a(scores.size(), false);
    std::vector<bool> used_b(scores.empty() ? 0 : scores[0].size(), false);

    for (const auto& [score, peak_a, peak_b] : candidates)
    {
      if (!used_a[peak_a] && !used_b[peak_b])
      {
        connections[link_idx].push_back({peak_a, peak_b});
        used_a[peak_a] = true;
        used_b[peak_b] = true;
        assigned_connections++;

#if 0
        // Debug assigned connections
        if (link_idx < 5)
        {
          qDebug() << "  Assigned connection:" << peak_a << "->" << peak_b
                   << "with score" << score;
        }
#endif
        // Limit connections per link to prevent over-connection
        if (connections[link_idx].size() >= 10)
          break;
      }
    }

#if 0
    if (link_idx < 5)
    {
      qDebug() << "  Link" << link_idx
               << "final connections:" << connections[link_idx].size();
    }
#endif
  }

#if 0
  qDebug() << "AssignGreedy summary: Total candidates:" << total_candidates
           << "Assigned connections:" << assigned_connections;
#endif

  return connections;
}

inline std::vector<Person> TRT_pose::connectParts(
    const std::vector<std::vector<Peak>>& peaks,
    const std::vector<std::vector<std::array<int, 2>>>& connections)
{

  std::vector<Person> persons;
  const auto& topology = TrtTopology::getCOCOTopology();

  // Debug: count total connections
  int total_connections = 0;
  for (const auto& link_conns : connections)
  {
    total_connections += link_conns.size();
  }
#if 0
  qDebug() << "ConnectParts: Starting with" << total_connections
           << "connections";
#endif
  // Build adjacency graph
  std::vector<std::vector<std::vector<int>>> adj_graph(18);
  for (int part = 0; part < 18; part++)
  {
    adj_graph[part].resize(peaks[part].size());
  }

  // Add connections to graph
  for (size_t link_idx = 0; link_idx < connections.size(); link_idx++)
  {
    const auto& link = topology[link_idx];
    int part_a = link[2];
    int part_b = link[3];

    for (const auto& conn : connections[link_idx])
    {
      adj_graph[part_a][conn[0]].push_back(part_b * 1000 + conn[1]);
      adj_graph[part_b][conn[1]].push_back(part_a * 1000 + conn[0]);
    }
  }

  // Find connected components using DFS
  std::vector<std::vector<bool>> visited(18);
  for (int part = 0; part < 18; part++)
  {
    visited[part].resize(peaks[part].size(), false);
  }

  for (int part = 0; part < 18; part++)
  {
    for (size_t peak_idx = 0; peak_idx < peaks[part].size(); peak_idx++)
    {
      if (!visited[part][peak_idx])
      {
        Person person;
        for (auto& kp : person.keypoints)
        {
          kp.x = kp.y = kp.confidence = 0.0f;
        }

        // DFS to find all connected parts
        std::vector<std::pair<int, int>> stack;
        stack.push_back({part, static_cast<int>(peak_idx)});

        float total_confidence = 0.0f;
        int valid_keypoints = 0;

        while (!stack.empty())
        {
          auto [curr_part, curr_peak] = stack.back();
          stack.pop_back();

          if (visited[curr_part][curr_peak])
            continue;
          visited[curr_part][curr_peak] = true;

          // Skip if this body part already has a keypoint assigned
          if (person.keypoints[curr_part].confidence > 0.0f)
          {
            // Keep the one with higher confidence
            const auto& peak = peaks[curr_part][curr_peak];
            if (peak.confidence > person.keypoints[curr_part].confidence)
            {
              total_confidence
                  -= person.keypoints[curr_part].confidence; // Remove old
              person.keypoints[curr_part]
                  = {static_cast<float>(peak.x),
                     static_cast<float>(peak.y),
                     peak.confidence};
              total_confidence += peak.confidence; // Add new
            }
            // Don't increment valid_keypoints since we're replacing
          }
          else
          {
            // First keypoint for this body part
            const auto& peak = peaks[curr_part][curr_peak];
            person.keypoints[curr_part]
                = {static_cast<float>(peak.x),
                   static_cast<float>(peak.y),
                   peak.confidence};
            total_confidence += peak.confidence;
            valid_keypoints++;
          }

          // Add connected peaks to stack
          for (int neighbor : adj_graph[curr_part][curr_peak])
          {
            int neighbor_part = neighbor / 1000;
            int neighbor_peak = neighbor % 1000;
            if (!visited[neighbor_part][neighbor_peak])
            {
              stack.push_back({neighbor_part, neighbor_peak});
            }
          }
        }

        // Filter out overly large connected components (likely multiple people merged)
        if (valid_keypoints >= 3 && valid_keypoints <= 18)
        { // Reasonable range for single person
          person.total_score = total_confidence / valid_keypoints;
          // Only add if score is above minimum threshold
          float score_threshold = config_.confidence_threshold * 1.5f;
          if (person.total_score > score_threshold)
          {
            // Additional check: ensure keypoints are spatially coherent
            if (isSpatiallyCoherent(person))
            {
              persons.push_back(person);
#if 0
              qDebug() << "Added person with" << valid_keypoints
                       << "keypoints, score:" << person.total_score;
#endif
            }
            else
            {
#if 0
              qDebug()
                  << "Rejected person - not spatially coherent, keypoints:"
                  << valid_keypoints;
#endif
            }
          }
          else
          {
#if 0
            qDebug() << "Rejected person - low score:" << person.total_score
                     << "threshold:" << score_threshold;
#endif
          }
        }
        else
        {
#if 0
          qDebug() << "Rejected person - invalid keypoint count:"
                   << valid_keypoints;
#endif
        }
      }
    }
  }

#if 0
  qDebug() << "ConnectParts: Final output -" << persons.size() << "persons";
#endif

  return persons;
}

inline std::vector<Person> TRT_pose::processOutput(
    const float* cmap_data,
    const float* paf_data,
    int height,
    int width)
{

  // Debug: Check peak detection for nose (part 0)
  // debugPeakDetection(cmap_data, height, width, 0);

  // 1. Find peaks in confidence maps
  auto peaks = findPeaks(cmap_data, height, width);

  // 2. Refine peak locations
  refinePeaks(peaks, cmap_data, height, width);

  // 3. Score PAF connections (using the fixed scoring with proper scaling)
  auto paf_scores = scorePAF(peaks, paf_data, height, width);

  // 4. Assign connections using Hungarian algorithm
  auto connections = assignConnections(paf_scores, peaks);

  // 5. Connect parts to form persons
  auto persons = connectParts(peaks, connections);

  // 6. Apply Non-Maximum Suppression to eliminate overlapping persons
  auto filtered_persons = applyNMS(persons, 0.5f); // 50% overlap threshold

  return filtered_persons;
}

inline QImage TRT_pose::visualizePoses(
    const std::vector<Person>& persons,
    int img_width,
    int img_height,
    int model_width,
    int model_height)
{
  QImage vis_image(img_width, img_height, QImage::Format_RGBA8888);
  vis_image.fill(Qt::black);

  QPainter painter(&vis_image);
  painter.setRenderHint(QPainter::Antialiasing);

  // Draw skeleton connections (official TRT-Pose format)
  const std::vector<std::array<int, 2>> skeleton = {
      {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, // legs and hips
      {5, 7},   {6, 8},   {7, 9},   {8, 10},            // arms
      {1, 2},   {0, 1},   {0, 2},   {1, 3},   {2, 4},   // face
      {3, 5},   {4, 6},                                 // ears to shoulders
      {17, 0},  {17, 5},  {17, 6},  {17, 11}, {17, 12}  // neck connections
  };

  for (const auto& person : persons)
  {
    // Draw connections
    painter.setPen(QPen(Qt::green, 2));
    for (const auto& connection : skeleton)
    {
      const auto& kp1 = person.keypoints[connection[0]];
      const auto& kp2 = person.keypoints[connection[1]];

      if (kp1.confidence > config_.confidence_threshold
          && kp2.confidence > config_.confidence_threshold)
      {
        float x1 = kp1.x * img_width / model_width;
        float y1 = kp1.y * img_height / model_height;
        float x2 = kp2.x * img_width / model_width;
        float y2 = kp2.y * img_height / model_height;

        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2));
      }
    }

    // Draw keypoints
    painter.setPen(QPen(Qt::red, 1));
    painter.setBrush(QBrush(Qt::red));
    for (const auto& kp : person.keypoints)
    {
      if (kp.confidence > config_.confidence_threshold)
      {
        float x = kp.x * img_width / model_width;
        float y = kp.y * img_height / model_height;
        painter.drawEllipse(QPointF(x, y), 3, 3);
      }
    }
  }

  return vis_image;
}

inline std::vector<Person>
TRT_pose::applyNMS(const std::vector<Person>& persons, float overlap_threshold)
{
  if (persons.empty())
    return persons;

  // Calculate bounding box for each person
  std::vector<std::array<float, 4>> bboxes; // [x_min, y_min, x_max, y_max]
  std::vector<float> scores;

  for (const auto& person : persons)
  {
    float x_min = 10000, y_min = 10000, x_max = -10000, y_max = -10000;
    int valid_kpts = 0;

    for (const auto& kp : person.keypoints)
    {
      if (kp.confidence > config_.confidence_threshold)
      {
        x_min = std::min(x_min, kp.x);
        y_min = std::min(y_min, kp.y);
        x_max = std::max(x_max, kp.x);
        y_max = std::max(y_max, kp.y);
        valid_kpts++;
      }
    }

    if (valid_kpts >= 3)
    {
      bboxes.push_back({x_min, y_min, x_max, y_max});
      scores.push_back(person.total_score);
    }
  }

  // Sort by score (descending)
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(),
      indices.end(),
      [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<bool> suppressed(indices.size(), false);
  std::vector<Person> filtered_persons;

  for (size_t i = 0; i < indices.size(); i++)
  {
    int idx = indices[i];
    if (suppressed[idx])
      continue;

    filtered_persons.push_back(persons[idx]);

    // Suppress overlapping detections
    for (size_t j = i + 1; j < indices.size(); j++)
    {
      int idx2 = indices[j];
      if (suppressed[idx2])
        continue;

      // Calculate IoU
      float x1 = std::max(bboxes[idx][0], bboxes[idx2][0]);
      float y1 = std::max(bboxes[idx][1], bboxes[idx2][1]);
      float x2 = std::min(bboxes[idx][2], bboxes[idx2][2]);
      float y2 = std::min(bboxes[idx][3], bboxes[idx2][3]);

      if (x1 < x2 && y1 < y2)
      {
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = (bboxes[idx][2] - bboxes[idx][0])
                      * (bboxes[idx][3] - bboxes[idx][1]);
        float area2 = (bboxes[idx2][2] - bboxes[idx2][0])
                      * (bboxes[idx2][3] - bboxes[idx2][1]);
        float union_area = area1 + area2 - intersection;

        if (intersection / union_area > overlap_threshold)
        {
          suppressed[idx2] = true;
        }
      }
    }
  }

  return filtered_persons;
}

inline QImage TRT_pose::visualizeConfidenceMaps(
    const float* cmap_data,
    int cmap_height,
    int cmap_width,
    int img_width,
    int img_height,
    float max_confidence,
    const std::vector<int>& parts_to_show)
{

  QImage overlay(img_width, img_height, QImage::Format_RGBA8888);
  overlay.fill(QColor(0, 0, 0, 0)); // Transparent

  QPainter painter(&overlay);
  painter.setRenderHint(QPainter::Antialiasing);

  // Scale factors for mapping confidence map to image coordinates
  float scale_x = static_cast<float>(img_width) / cmap_width;
  float scale_y = static_cast<float>(img_height) / cmap_height;

  // Define colors for different body parts
  std::vector<QColor> part_colors = {
      QColor(255, 0, 0),     // 0: nose - red
      QColor(255, 128, 0),   // 1: left_eye - orange
      QColor(255, 255, 0),   // 2: right_eye - yellow
      QColor(128, 255, 0),   // 3: left_ear - lime
      QColor(0, 255, 0),     // 4: right_ear - green
      QColor(0, 255, 128),   // 5: left_shoulder - cyan-green
      QColor(0, 255, 255),   // 6: right_shoulder - cyan
      QColor(0, 128, 255),   // 7: left_elbow - light blue
      QColor(0, 0, 255),     // 8: right_elbow - blue
      QColor(128, 0, 255),   // 9: left_wrist - purple
      QColor(255, 0, 255),   // 10: right_wrist - magenta
      QColor(255, 0, 128),   // 11: left_hip - pink
      QColor(255, 128, 128), // 12: right_hip - light pink
      QColor(128, 255, 128), // 13: left_knee - light green
      QColor(128, 128, 255), // 14: right_knee - light blue
      QColor(64, 255, 64),   // 15: left_ankle - dark green
      QColor(64, 64, 255),   // 16: right_ankle - dark blue
      QColor(255, 255, 255)  // 17: neck - white
  };

  // Determine which parts to visualize
  std::vector<int> parts_list;
  if (parts_to_show.empty())
  {
    // Show all 18 keypoint parts
    parts_list
        = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  }
  else
  {
    parts_list = parts_to_show;
  }

  // Render each selected part's confidence map
  for (int part : parts_list)
  {
    if (part < 0 || part >= 18)
      continue;

    const float* part_map = cmap_data + part * cmap_height * cmap_width;
    QColor part_color = part_colors[part % part_colors.size()];

    // Find threshold for this part (show top 10% of confidence values)
    float part_threshold = max_confidence * 0.1f;

    for (int h = 0; h < cmap_height; h++)
    {
      for (int w = 0; w < cmap_width; w++)
      {
        float confidence = part_map[h * cmap_width + w];

        if (confidence > part_threshold)
        {
          // Calculate alpha based on confidence (0-128 for semi-transparency)
          int alpha = static_cast<int>(128 * confidence / max_confidence);
          alpha = std::clamp(alpha, 0, 128);

          QColor pixel_color = part_color;
          pixel_color.setAlpha(alpha);

          // Map to image coordinates and draw
          int img_x = static_cast<int>(w * scale_x);
          int img_y = static_cast<int>(h * scale_y);
          int size_x = std::max(1, static_cast<int>(scale_x));
          int size_y = std::max(1, static_cast<int>(scale_y));

          painter.fillRect(img_x, img_y, size_x, size_y, pixel_color);
        }
      }
    }
  }

  // Add legend
  painter.setPen(QPen(Qt::white, 2));
  painter.setFont(QFont("Arial", 10));
  int legend_y = 10;

  for (size_t i = 0; i < parts_list.size(); i++)
  {
    int part = parts_list[i];
    if (part >= 0
        && part < static_cast<int>(TrtTopology::getKeypointNames().size()))
    {
      QColor legend_color = part_colors[part % part_colors.size()];
      legend_color.setAlpha(255);

      // Draw color square
      painter.fillRect(10, legend_y, 15, 15, legend_color);
      painter.drawRect(10, legend_y, 15, 15);

      // Draw part name
      QString part_name
          = QString::fromStdString(TrtTopology::getKeypointNames()[part]);
      painter.drawText(30, legend_y + 12, part_name);

      legend_y += 20;
    }
  }

  return overlay;
}

inline QImage TRT_pose::visualizePAF(
    const float* paf_data,
    int paf_height,
    int paf_width,
    int img_width,
    int img_height,
    float scale_factor,
    float magnitude_threshold)
{

  QImage paf_overlay(img_width, img_height, QImage::Format_RGBA8888);
  paf_overlay.fill(QColor(0, 0, 0, 0)); // Transparent

  QPainter painter(&paf_overlay);
  painter.setRenderHint(QPainter::Antialiasing);

  // Scale factors for mapping PAF coordinates to image coordinates
  float scale_x = static_cast<float>(img_width) / paf_width;
  float scale_y = static_cast<float>(img_height) / paf_height;

  // Define colors for different PAF channels (21 limb connections)
  std::vector<QColor> paf_colors = {
      QColor(255, 0, 0, 128),     // 0-1: neck -> nose
      QColor(255, 128, 0, 128),   // 2-3: nose -> left_eye
      QColor(255, 255, 0, 128),   // 4-5: nose -> right_eye
      QColor(128, 255, 0, 128),   // 6-7: left_eye -> left_ear
      QColor(0, 255, 0, 128),     // 8-9: right_eye -> right_ear
      QColor(0, 255, 128, 128),   // 10-11: neck -> left_shoulder
      QColor(0, 255, 255, 128),   // 12-13: neck -> right_shoulder
      QColor(0, 128, 255, 128),   // 14-15: left_shoulder -> left_elbow
      QColor(0, 0, 255, 128),     // 16-17: left_elbow -> left_wrist
      QColor(128, 0, 255, 128),   // 18-19: right_shoulder -> right_elbow
      QColor(255, 0, 255, 128),   // 20-21: right_elbow -> right_wrist
      QColor(255, 0, 128, 128),   // 22-23: neck -> left_hip
      QColor(255, 128, 128, 128), // 24-25: neck -> right_hip
      QColor(128, 255, 128, 128), // 26-27: left_hip -> left_knee
      QColor(128, 128, 255, 128), // 28-29: left_knee -> left_ankle
      QColor(64, 255, 64, 128),   // 30-31: right_hip -> right_knee
      QColor(64, 64, 255, 128),   // 32-33: right_knee -> right_ankle
      QColor(255, 255, 128, 128), // 34-35: left_shoulder -> right_shoulder
      QColor(128, 255, 255, 128), // 36-37: left_hip -> right_hip
      QColor(255, 128, 255, 128), // 38-39: left_shoulder -> left_hip
      QColor(128, 128, 128, 128)  // 40-41: right_shoulder -> right_hip
  };

  // Draw PAF vectors as arrows
  int step = static_cast<int>(scale_factor);
  for (int link_idx = 0; link_idx < 21; link_idx++)
  {
    int x_channel = link_idx * 2;
    int y_channel = link_idx * 2 + 1;

    if (x_channel >= 42 || y_channel >= 42)
      continue; // Safety check

    const float* paf_x = paf_data + x_channel * paf_height * paf_width;
    const float* paf_y = paf_data + y_channel * paf_height * paf_width;

    QColor link_color = paf_colors[link_idx % paf_colors.size()];
    painter.setPen(QPen(link_color, 1));

    for (int h = 0; h < paf_height; h += step)
    {
      for (int w = 0; w < paf_width; w += step)
      {
        float vx = paf_x[h * paf_width + w];
        float vy = paf_y[h * paf_width + w];

        // Calculate vector magnitude
        float magnitude = std::sqrt(vx * vx + vy * vy);

        if (magnitude > magnitude_threshold)
        {
          // Map to image coordinates
          float img_x = w * scale_x;
          float img_y = h * scale_y;

          // Scale vector length for visibility
          float arrow_length = magnitude * 20.0f; // Amplify for visibility
          float end_x = img_x + vx * arrow_length;
          float end_y = img_y + vy * arrow_length;

          // Draw vector as line with arrowhead
          QPointF start(img_x, img_y);
          QPointF end(end_x, end_y);

          painter.drawLine(start, end);

          // Draw simple arrowhead
          if (arrow_length > 5.0f)
          {
            float angle = std::atan2(vy, vx);
            float arrow_size = 3.0f;

            QPointF arrow1(
                end_x - arrow_size * std::cos(angle - M_PI / 6),
                end_y - arrow_size * std::sin(angle - M_PI / 6));
            QPointF arrow2(
                end_x - arrow_size * std::cos(angle + M_PI / 6),
                end_y - arrow_size * std::sin(angle + M_PI / 6));

            painter.drawLine(end, arrow1);
            painter.drawLine(end, arrow2);
          }
        }
      }
    }
  }

  return paf_overlay;
}

inline QImage TRT_pose::visualizePeaks(
    const std::vector<std::vector<Peak>>& peaks,
    int img_width,
    int img_height,
    int model_width,
    int model_height)
{
  QImage peaks_overlay(img_width, img_height, QImage::Format_RGBA8888);
  peaks_overlay.fill(QColor(0, 0, 0, 0)); // Transparent

  QPainter painter(&peaks_overlay);
  painter.setRenderHint(QPainter::Antialiasing);

  // Define colors for different body parts (same as confidence map colors)
  std::vector<QColor> part_colors = {
      QColor(255, 0, 0),     // 0: nose - red
      QColor(255, 128, 0),   // 1: left_eye - orange
      QColor(255, 255, 0),   // 2: right_eye - yellow
      QColor(128, 255, 0),   // 3: left_ear - lime
      QColor(0, 255, 0),     // 4: right_ear - green
      QColor(0, 255, 128),   // 5: left_shoulder - cyan-green
      QColor(0, 255, 255),   // 6: right_shoulder - cyan
      QColor(0, 128, 255),   // 7: left_elbow - light blue
      QColor(0, 0, 255),     // 8: right_elbow - blue
      QColor(128, 0, 255),   // 9: left_wrist - purple
      QColor(255, 0, 255),   // 10: right_wrist - magenta
      QColor(255, 0, 128),   // 11: left_hip - pink
      QColor(255, 128, 128), // 12: right_hip - light pink
      QColor(128, 255, 128), // 13: left_knee - light green
      QColor(128, 128, 255), // 14: right_knee - light blue
      QColor(64, 255, 64),   // 15: left_ankle - dark green
      QColor(64, 64, 255),   // 16: right_ankle - dark blue
      QColor(255, 255, 255)  // 17: neck - white
  };

  // Draw detected peaks for each part
  for (int part = 0; part < 18; part++)
  {
    const auto& part_peaks = peaks[part];
    if (part_peaks.empty())
      continue;

    QColor part_color = part_colors[part % part_colors.size()];

    for (size_t i = 0; i < part_peaks.size(); i++)
    {
      const auto& peak = part_peaks[i];

      // Convert from model coordinates to image coordinates
      float img_x = peak.x * img_width / model_width;
      float img_y = peak.y * img_height / model_height;

      // Draw peak as a circle with size proportional to confidence
      float radius = 3.0f + (peak.confidence * 5.0f); // 3-8 pixel radius
      radius = std::clamp(radius, 3.0f, 10.0f);

      // Draw outer circle (white border for visibility)
      painter.setPen(QPen(Qt::white, 2));
      painter.setBrush(QBrush(part_color));
      painter.drawEllipse(QPointF(img_x, img_y), radius, radius);

      // Draw part number inside the circle for identification
      painter.setPen(QPen(Qt::black, 1));
      painter.setFont(QFont("Arial", 8, QFont::Bold));
      painter.drawText(
          QRectF(img_x - radius, img_y - radius, radius * 2, radius * 2),
          Qt::AlignCenter,
          QString::number(part));

      // Debug: show peak info for first few peaks
      if (i == 0 && (part == 0 || part == 17)) // nose or neck
      {
        qDebug() << "Peak visualization: Part" << part << "at model coords ("
                 << peak.x << "," << peak.y << ")"
                 << "-> image coords (" << img_x << "," << img_y << ")"
                 << "confidence:" << peak.confidence;
      }
    }
  }

  // Add legend showing part numbers and colors
  painter.setPen(QPen(Qt::white, 2));
  painter.setFont(QFont("Arial", 10));
  int legend_x = img_width - 200;
  int legend_y = 10;

  painter.fillRect(legend_x - 5, legend_y - 5, 190, 380, QColor(0, 0, 0, 128));
  painter.drawText(legend_x, legend_y + 15, "Detected Peaks:");

  for (int part = 0; part < 18; part++)
  {
    const auto& part_peaks = peaks[part];
    if (part_peaks.empty())
      continue;

    QColor legend_color = part_colors[part % part_colors.size()];

    int y_pos = legend_y + 30 + part * 20;

    // Draw color circle
    painter.setPen(QPen(Qt::white, 1));
    painter.setBrush(QBrush(legend_color));
    painter.drawEllipse(legend_x, y_pos, 12, 12);

    // Draw part info
    painter.setPen(QPen(Qt::white, 1));
    QString part_name
        = QString::fromStdString(TrtTopology::getKeypointNames()[part]);
    painter.drawText(
        legend_x + 20,
        y_pos + 10,
        QString("%1: %2 (%3)")
            .arg(part)
            .arg(part_name)
            .arg(part_peaks.size()));
  }

  return peaks_overlay;
}

inline QImage TRT_pose::visualizePAFScores(
    const std::vector<std::vector<Peak>>& peaks,
    const std::vector<std::vector<std::vector<float>>>& paf_scores,
    int img_width,
    int img_height,
    int model_width,
    int model_height,
    float score_threshold)
{
  QImage paf_scores_overlay(img_width, img_height, QImage::Format_RGBA8888);
  paf_scores_overlay.fill(QColor(0, 0, 0, 0)); // Transparent

  QPainter painter(&paf_scores_overlay);
  painter.setRenderHint(QPainter::Antialiasing);

  const auto& topology = TrtTopology::getCOCOTopology();

  // Define colors for different connections (similar to PAF visualization)
  std::vector<QColor> connection_colors = {
      QColor(255, 0, 0),     // 0: neck -> nose - red
      QColor(255, 128, 0),   // 1: nose -> left_eye - orange
      QColor(255, 255, 0),   // 2: nose -> right_eye - yellow
      QColor(128, 255, 0),   // 3: left_eye -> left_ear - lime
      QColor(0, 255, 0),     // 4: right_eye -> right_ear - green
      QColor(0, 255, 128),   // 5: neck -> left_shoulder - cyan-green
      QColor(0, 255, 255),   // 6: neck -> right_shoulder - cyan
      QColor(0, 128, 255),   // 7: left_shoulder -> left_elbow - light blue
      QColor(0, 0, 255),     // 8: left_elbow -> left_wrist - blue
      QColor(128, 0, 255),   // 9: right_shoulder -> right_elbow - purple
      QColor(255, 0, 255),   // 10: right_elbow -> right_wrist - magenta
      QColor(255, 0, 128),   // 11: neck -> left_hip - pink
      QColor(255, 128, 128), // 12: neck -> right_hip - light pink
      QColor(128, 255, 128), // 13: left_hip -> left_knee - light green
      QColor(128, 128, 255), // 14: left_knee -> left_ankle - light blue
      QColor(64, 255, 64),   // 15: right_hip -> right_knee - dark green
      QColor(64, 64, 255),   // 16: right_knee -> right_ankle - dark blue
      QColor(
          255, 255, 128), // 17: left_shoulder -> right_shoulder - yellow-white
      QColor(128, 255, 255), // 18: left_hip -> right_hip - cyan-white
      QColor(255, 128, 255), // 19: left_shoulder -> left_hip - magenta-white
      QColor(128, 128, 128)  // 20: right_shoulder -> right_hip - gray
  };

  int total_connections = 0;
  int good_connections = 0;

  // Draw PAF score connections
  for (size_t link_idx = 0;
       link_idx < paf_scores.size() && link_idx < topology.size();
       link_idx++)
  {
    const auto& link = topology[link_idx];
    int part_a = link[2];
    int part_b = link[3];

    const auto& scores = paf_scores[link_idx];
    const auto& peaks_a = peaks[part_a];
    const auto& peaks_b = peaks[part_b];

    if (scores.empty() || peaks_a.empty() || peaks_b.empty())
      continue;

    QColor link_color = connection_colors[link_idx % connection_colors.size()];

    // Find best score for this link to normalize visualization
    float max_score = 0.0f;
    for (size_t i = 0; i < scores.size(); i++)
    {
      for (size_t j = 0; j < scores[i].size(); j++)
      {
        max_score = std::max(max_score, scores[i][j]);
      }
    }

    // Draw all connections above threshold
    for (size_t i = 0; i < scores.size() && i < peaks_a.size(); i++)
    {
      for (size_t j = 0; j < scores[i].size() && j < peaks_b.size(); j++)
      {
        float score = scores[i][j];
        total_connections++;

        if (score > score_threshold)
        {
          good_connections++;

          // Convert peak coordinates to image coordinates
          float x1 = peaks_a[i].x * img_width / model_width;
          float y1 = peaks_a[i].y * img_height / model_height;
          float x2 = peaks_b[j].x * img_width / model_width;
          float y2 = peaks_b[j].y * img_height / model_height;

          // Line thickness and alpha based on score strength
          float normalized_score = (max_score > 0) ? score / max_score : 0.0f;
          int line_width
              = std::max(1, static_cast<int>(normalized_score * 4.0f));
          int alpha
              = std::max(64, static_cast<int>(normalized_score * 255.0f));

          QColor line_color = link_color;
          line_color.setAlpha(alpha);

          painter.setPen(QPen(line_color, line_width));
          painter.drawLine(QPointF(x1, y1), QPointF(x2, y2));

          // Draw score value for strong connections
          if (score > score_threshold * 2.0f)
          {
            float mid_x = (x1 + x2) / 2.0f;
            float mid_y = (y1 + y2) / 2.0f;

            painter.setPen(QPen(Qt::white, 1));
            painter.setFont(QFont("Arial", 8));
            painter.drawText(
                QPointF(mid_x, mid_y), QString::number(score, 'f', 2));
          }

          // Debug logging for first few good connections
          if (good_connections <= 5
              && (part_a == 0 || part_a == 17 || part_b == 0 || part_b == 17))
          {
            qDebug() << "PAF connection" << good_connections << ":"
                     << "parts" << part_a << "->" << part_b << "peaks ("
                     << peaks_a[i].x << "," << peaks_a[i].y << ") -> ("
                     << peaks_b[j].x << "," << peaks_b[j].y << ")"
                     << "score:" << score;
          }
        }
      }
    }
  }

  // Add legend showing connection statistics
  painter.setPen(QPen(Qt::white, 2));
  painter.setFont(QFont("Arial", 12, QFont::Bold));

  int legend_x = 10;
  int legend_y = img_height - 100;

  painter.fillRect(legend_x - 5, legend_y - 5, 300, 90, QColor(0, 0, 0, 128));
  painter.drawText(legend_x, legend_y + 15, "PAF Score Analysis:");
  painter.drawText(
      legend_x,
      legend_y + 35,
      QString("Total connections tested: %1").arg(total_connections));
  painter.drawText(
      legend_x,
      legend_y + 55,
      QString("Good connections (>%1): %2")
          .arg(score_threshold)
          .arg(good_connections));
  painter.drawText(
      legend_x,
      legend_y + 75,
      QString("Success rate: %1%")
          .arg(
              total_connections > 0
                  ? (good_connections * 100 / total_connections)
                  : 0));

  qDebug() << "PAF Score Summary: Total connections:" << total_connections
           << "Good connections:" << good_connections
           << "Threshold:" << score_threshold;

  return paf_scores_overlay;
}

inline bool TRT_pose::isSpatiallyCoherent(const Person& person)
{
  // Calculate bounding box of valid keypoints
  float min_x = 1000, max_x = -1000, min_y = 1000, max_y = -1000;
  int valid_count = 0;

  for (const auto& kp : person.keypoints)
  {
    if (kp.confidence > config_.confidence_threshold)
    {
      min_x = std::min(min_x, kp.x);
      max_x = std::max(max_x, kp.x);
      min_y = std::min(min_y, kp.y);
      max_y = std::max(max_y, kp.y);
      valid_count++;
    }
  }

  if (valid_count < 3)
  {
    // qDebug() << "SpatialCoherence: Too few keypoints:" << valid_count;
    return false;
  }

  // Check if bounding box is reasonable for a single person
  float width = max_x - min_x;
  float height = max_y - min_y;
#if 0
  qDebug() << "SpatialCoherence: Bounding box - width:" << width
           << "height:" << height << "bounds: (" << min_x << "," << min_y
           << ") to (" << max_x << "," << max_y << ")";
#endif
  // Reject if bounding box is too large (likely multiple people)
  // Note: These are in model coordinates (128x128), not image coordinates!
  // A person typically spans 30-80 pixels in a 128x128 model
  if (width > 80.0f || height > 100.0f)
  {
    // qDebug() << "  Rejected: Bounding box too large";
    return false;
  }

  // Reject if bounding box is too small (likely noise)
  if (width < 5.0f || height < 10.0f)
  {
    // qDebug() << "  Rejected: Bounding box too small";
    return false;
  }
#if 0
  qDebug() << "  Accepted: Spatial coherence OK";
#endif
  return true;
}

inline void TRT_pose::debugPeakDetection(
    const float* cmap_data,
    int height,
    int width,
    int part_id)
{
  if (part_id < 0 || part_id >= 18)
    return;

  const float* part_map = cmap_data + part_id * height * width;

  // Find true maximum
  float max_val = 0.0f;
  int max_h = 0, max_w = 0;
  for (int h = 0; h < height; h++)
  {
    for (int w = 0; w < width; w++)
    {
      float val = part_map[h * width + w];
      if (val > max_val)
      {
        max_val = val;
        max_h = h;
        max_w = w;
      }
    }
  }

  qDebug() << "Part" << part_id << "true max:" << max_val
           << "at model coords (" << max_w << "," << max_h << ")";

  // Show what peaks we actually detect
  std::vector<Peak> detected_peaks;
  for (int h = config_.peak_window_size; h < height - config_.peak_window_size;
       h++)
  {
    for (int w = config_.peak_window_size;
         w < width - config_.peak_window_size;
         w++)
    {
      float val = part_map[h * width + w];
      if (val > config_.confidence_threshold
          && isLocalMaximum(
              part_map, h, w, height, width, config_.confidence_threshold))
      {
        detected_peaks.push_back(
            {static_cast<float>(w), static_cast<float>(h), val, 0});
        if (detected_peaks.size() <= 3)
        { // Show first few
          qDebug() << "  Detected peak:" << val << "at model coords (" << w
                   << "," << h << ")";
        }
      }
    }
  }
  qDebug() << "  Total detected peaks:" << detected_peaks.size();
}

} // namespace Onnx
