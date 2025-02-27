#pragma once
#include <QDebug>

#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>

namespace OnnxModels::RTMPose
{

struct RTMPose_fullbody
{
  static constexpr int NUM_KPS = 26;

  struct keypoint
  {
    float x, y;
    float confidence() const noexcept { return 1.0f; }
  };

  struct pose_data
  {
    keypoint keypoints[NUM_KPS]{};
  };

  // Decode the two simcc output arrays into a pose_data
  static bool decode(
      std::span<Ort::Value> outputTensor,
      std::optional<RTMPose_fullbody::pose_data>& out)
  {
    out.reset();
    if (outputTensor.size() == 0)
      return false;

    const auto& outputTensor_x = outputTensor[0];
    const auto& outputTensor_y = outputTensor[1];

    const int Nfloats_x
        = outputTensor_x.GetTensorTypeAndShapeInfo().GetElementCount();
    const int Nfloats_y
        = outputTensor_y.GetTensorTypeAndShapeInfo().GetElementCount();
    qDebug() << "SimCC x float count:" << Nfloats_x
             << "SimCC y float count:" << Nfloats_y;

    // We expect 26 keypoints
    if (Nfloats_x != 26 * 384 || Nfloats_y != 26 * 512)
      return false;

    const float* data_x = outputTensor_x.GetTensorData<float>();
    const float* data_y = outputTensor_y.GetTensorData<float>();

    pose_data decoded{};

    // loop through the 26 keypoints
    for (int k = 0; k < NUM_KPS; k++)
    {

      // 384 bins for x
      const int bin_x = 384;

      // find simccx max, get index
      const float* row_x = data_x + (k * bin_x);
      float maxValX = row_x[0];
      int maxIdxX = 0;
      for (int i = 1; i < bin_x; i++)
      {
        if (row_x[i] > maxValX)
        {
          maxValX = row_x[i];
          maxIdxX = i;
        }
      }
      qDebug() << "Keypoint" << k << ": maxIdxX =" << maxIdxX
               << ", maxValX =" << maxValX;

      // 512 bins for y
      const int bin_y = 512;

      // find simccy max, get index
      const float* row_y = data_y + (k * bin_y);
      float maxValY = row_y[0];
      int maxIdxY = 0;
      for (int i = 1; i < bin_y; i++)
      {
        if (row_y[i] > maxValY)
        {
          maxValY = row_y[i];
          maxIdxY = i;
        }
      }
      qDebug() << "Keypoint" << k << ": maxIdxY =" << maxIdxY
               << ", maxValY =" << maxValY;

      // convert simccx & simccy indexes to 192 x 256 img coordinates
      float xCoordinate = (maxIdxX / float(bin_x - 1)) * 192;
      float yCoordinate = (maxIdxY / float(bin_y - 1)) * 256;

      // Fill struct with coordinates!!
      decoded.keypoints[k].x = xCoordinate;
      decoded.keypoints[k].y = yCoordinate;

      qDebug() << "Keypoint" << k << ": xCoord =" << xCoordinate
               << ", yCoord =" << yCoordinate;
    }

    out = decoded;
    return true;
  }
};
}
