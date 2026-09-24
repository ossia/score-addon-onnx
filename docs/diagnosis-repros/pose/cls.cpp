#include <Onnx/helpers/ModelRole.hpp>
#include <cstdio>
using namespace Onnx;
static const char* K[]={"Unknown","BlazePoseDetector","PalmDetector","BlazeFaceDetector","PersonDetector","MultiClassDetector","YoloxDetector","RetinaFaceDetector","FaceBoxesDetector","BlazePoseLandmark","HandLandmark","FaceMeshLandmark","MobileFaceNet","SimccPose","HeatmapPose","XyScoreLandmark","YoloPose","RtmoPose","MoveNetPose","ReidEmbedder"};
static const char* D[]={"Unknown","Body","Hand","Face","Animal"};
void run(const char* n, ModelIO io){auto r=classify(io);std::printf("%-44s kind=%s domain=%s K=%d in=%dx%d\n",n,K[(int)r.kind],D[(int)r.domain],r.num_keypoints,r.input_w,r.input_h);}
int main(){
  run("FaceBoxesProd ORT_ENABLE_ALL (last dims 4/2)",{{{"input",{-1,3,-1,-1}}},{{"output",{-1,-1,4}},{"367",{-1,-1,2}}}});
  run("FaceBoxesProd ORT_DISABLE_ALL (all dynamic)",{{{"input",{-1,3,-1,-1}}},{{"output",{-1,-1,-1}},{"367",{-1,-1,-1}}}});
  run("rtmw3d_l_384x288 (dynamic K)",{{{"input",{-1,3,384,288}}},{{"simcc_x",{-1,-1,576}},{"simcc_y",{-1,-1,768}},{"simcc_z",{-1,-1,576}}}});
  run("dwpose-l-384x288 (dynamic K)",{{{"input",{-1,3,384,288}}},{{"simcc_x",{-1,-1,-1}},{"simcc_y",{-1,-1,-1}}}});
  run("animalpose hrnet_w32 256x256 K=20",{{{"img",{1,3,256,256}}},{{"heatmap",{1,20,64,64}}}});
  run("vitpose_s_aic 256x192 K=14",{{{"input",{-1,3,256,192}}},{{"output",{-1,14,64,48}}}});
  run("vitpose_s_mpii K=16",{{{"input",{-1,3,256,192}}},{{"output",{-1,16,64,48}}}});
  run("vitpose_s_coco_25 K=25",{{{"input",{-1,3,256,192}}},{{"output",{-1,25,64,48}}}});
  run("vitpose_s_ap10k 256x192 K=17",{{{"input",{-1,3,256,192}}},{{"output",{-1,17,64,48}}}});
  run("yolov8n-pose @320 [1,56,2100] (hypothetical)",{{{"images",{1,3,320,320}}},{{"output0",{1,56,2100}}}});
  run("yolov8 hand-pose K=21 @640 [1,68,8400] (hyp.)",{{{"images",{1,3,640,640}}},{{"output0",{1,68,8400}}}});
  run("yolov9 wholebody25 post (dynamic HxW)",{{{"input_bgr",{1,3,-1,-1}}},{{"batchno_classid_score_x1y1x2y2",{-1,7}}}});
  run("gold_yolo_n_head_post 480x640",{{{"input",{1,3,480,640}}},{{"batchno_classid_x1y1x2y2_score",{-1,7}}}});
  run("ailia yolox_tiny.opt 416",{{{"images",{1,3,416,416}}},{{"output",{1,3549,85}}}});
  run("wild2 blazepalm 256 NCHW",{{{"input",{1,3,256,256}}},{{"regressors",{1,2944,18}},{"classificators",{1,2944,1}}}});
}
