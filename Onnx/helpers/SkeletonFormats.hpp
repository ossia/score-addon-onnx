#pragma once
// Skeleton keypoint remapping: convert a model's native keypoint layout into a
// standard target layout (OpenPose, COCO, Halpe, ...). Qt/ossia-free.
//
// Each target joint is a weighted combination of source joints:
//   - direct(i)   : target joint IS source joint i           (permutation/subset)
//   - mid(i,j)    : target joint = (i + j) / 2                (synthesized, e.g.
//                   OpenPose Neck = mid(L_shoulder, R_shoulder))
//   - none()      : no source for this joint -> emitted as (0,0,0), confidence 0
//
// One direct table per (source, target) pair (no lossy hub): with only a couple
// of body sources this stays small AND lossless (e.g. BlazePose->BODY_25 keeps
// feet from its heel/foot_index landmarks, which a COCO-17 hub would drop).
//
// Index tables verified against primary sources (CMU OpenPose poseParameters,
// MediaPipe, COCO, AlphaPose Halpe-26). See docs/ research notes.
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Onnx::Skel
{

// Native keypoint layout a model produces.
enum class SourceSkeleton
{
  BlazePose33,
  Coco17,
  Hands21,
  FaceMesh468,
  Dlib68,
  Ap10k17,
  Mhr70 // InstantHMR / SAM-3D-Body keypoints (appended: keep values stable)
};

// Requested output layout (Native = no remap, handled by the caller).
enum class TargetSkeleton
{
  Native,
  Coco17,
  OpenPoseCoco18,
  OpenPoseBody25,
  Halpe26,
  Mpii16,
  H36m17,
  Dlib68,
  Hand21
};

// A target joint = sum of up to 4 weighted source joints. n==0 => unmappable.
// Weights need not lie in [0,1] or sum to 1: head_top extrapolates above the
// eyes via {eyeL, eyeR, nose} weights {1, 1, -1}.
struct JointMap
{
  int n;
  int idx[4];
  float w[4];
};
constexpr JointMap none() { return {0, {-1, -1, -1, -1}, {0, 0, 0, 0}}; }
constexpr JointMap direct(int i) { return {1, {i, -1, -1, -1}, {1.f, 0, 0, 0}}; }
constexpr JointMap mid(int i, int j)
{
  return {2, {i, j, -1, -1}, {0.5f, 0.5f, 0, 0}};
}
constexpr JointMap avg4(int i, int j, int k, int l)
{
  return {4, {i, j, k, l}, {0.25f, 0.25f, 0.25f, 0.25f}};
}
// Extrapolated point a + b - c (e.g. top-of-head ~= eyeL + eyeR - nose).
constexpr JointMap extrap(int a, int b, int c)
{
  return {3, {a, b, c, -1}, {1.f, 1.f, -1.f, 0}};
}

struct Bone
{
  int a, b;
};

// ============================ Source: COCO-17 ============================
// 0 nose 1 Leye 2 Reye 3 Lear 4 Rear 5 Lsh 6 Rsh 7 Lel 8 Rel 9 Lwr 10 Rwr
// 11 Lhip 12 Rhip 13 Lknee 14 Rknee 15 Lankle 16 Rankle

// COCO-17 -> OpenPose COCO-18 (synth Neck; right-before-left reorder).
inline constexpr JointMap coco17_to_coco18[] = {
    direct(0),  mid(5, 6),  direct(6),  direct(8),  direct(10), direct(5),
    direct(7),  direct(9),  direct(12), direct(14), direct(16), direct(11),
    direct(13), direct(15), direct(2),  direct(1),  direct(4),  direct(3)};

// COCO-17 -> OpenPose BODY_25 (synth Neck + MidHip; toes/heels unfillable).
inline constexpr JointMap coco17_to_body25[] = {
    direct(0),  mid(5, 6),  direct(6),  direct(8),  direct(10), direct(5),
    direct(7),  direct(9),  mid(11, 12), direct(12), direct(14), direct(16),
    direct(11), direct(13), direct(15), direct(2),  direct(1),  direct(4),
    direct(3),  none(),     none(),     none(),     none(),     none(),
    none()};

// COCO-17 -> Halpe-26 (first 17 == COCO order; synth Head/Neck/Hip; feet none).
inline constexpr JointMap coco17_to_halpe26[] = {
    direct(0),  direct(1),  direct(2),  direct(3),  direct(4),  direct(5),
    direct(6),  direct(7),  direct(8),  direct(9),  direct(10), direct(11),
    direct(12), direct(13), direct(14), direct(15), direct(16), mid(3, 4),
    mid(5, 6),  mid(11, 12), none(),    none(),     none(),     none(),
    none(),     none()};

// ============================ Source: BlazePose-33 ============================
// 0 nose 2 Leye 5 Reye 7 Lear 8 Rear 11 Lsh 12 Rsh 13 Lel 14 Rel 15 Lwr 16 Rwr
// 23 Lhip 24 Rhip 25 Lknee 26 Rknee 27 Lankle 28 Rankle 29 Lheel 30 Rheel
// 31 Lfoot_index 32 Rfoot_index

inline constexpr JointMap blaze33_to_coco17[] = {
    direct(0),  direct(2),  direct(5),  direct(7),  direct(8),  direct(11),
    direct(12), direct(13), direct(14), direct(15), direct(16), direct(23),
    direct(24), direct(25), direct(26), direct(27), direct(28)};

inline constexpr JointMap blaze33_to_coco18[] = {
    direct(0),  mid(11, 12), direct(12), direct(14), direct(16), direct(11),
    direct(13), direct(15), direct(24), direct(26), direct(28), direct(23),
    direct(25), direct(27), direct(5),  direct(2),  direct(8),  direct(7)};

// BlazePose -> BODY_25: feet filled from heel(29,30)/foot_index(31,32); only the
// two SmallToe slots stay unfillable.
inline constexpr JointMap blaze33_to_body25[] = {
    direct(0),  mid(11, 12), direct(12), direct(14), direct(16), direct(11),
    direct(13), direct(15), mid(23, 24), direct(24), direct(26), direct(28),
    direct(23), direct(25), direct(27), direct(5),  direct(2),  direct(8),
    direct(7),  direct(31), none(),      direct(29), direct(32), none(),
    direct(30)};

inline constexpr JointMap blaze33_to_halpe26[] = {
    direct(0),  direct(2),  direct(5),  direct(7),  direct(8),  direct(11),
    direct(12), direct(13), direct(14), direct(15), direct(16), direct(23),
    direct(24), direct(25), direct(26), direct(27), direct(28), mid(7, 8),
    mid(11, 12), mid(23, 24), direct(31), direct(32), none(),   none(),
    direct(29), direct(30)};

// ============================ body: MPII-16 / Human3.6M-17 ============================
// Best-effort: pelvis/thorax = hip/shoulder midpoints; upper_neck/neck blend
// nose toward the shoulders; head_top extrapolates above the eyes. Flagged
// approximate (torso/head joints have no exact COCO/BlazePose source).
// MPII-16: 0 RAnkle 1 RKnee 2 RHip 3 LHip 4 LKnee 5 LAnkle 6 Pelvis 7 Thorax
//          8 UpperNeck 9 HeadTop 10 RWrist 11 RElbow 12 RShoulder 13 LShoulder
//          14 LElbow 15 LWrist
inline constexpr JointMap coco17_to_mpii16[] = {
    direct(16), direct(14), direct(12), direct(11), direct(13), direct(15),
    mid(11, 12), mid(5, 6), avg4(0, 0, 5, 6), extrap(1, 2, 0), direct(10),
    direct(8), direct(6), direct(5), direct(7), direct(9)};
inline constexpr JointMap blaze33_to_mpii16[] = {
    direct(28), direct(26), direct(24), direct(23), direct(25), direct(27),
    mid(23, 24), mid(11, 12), avg4(0, 0, 11, 12), extrap(2, 5, 0), direct(16),
    direct(14), direct(12), direct(11), direct(13), direct(15)};

// Human3.6M-17: 0 Hip 1 RHip 2 RKnee 3 RFoot 4 LHip 5 LKnee 6 LFoot 7 Spine
//   8 Thorax 9 Neck 10 Head 11 LShoulder 12 LElbow 13 LWrist 14 RShoulder
//   15 RElbow 16 RWrist  (Spine = mean of hips+shoulders).
inline constexpr JointMap coco17_to_h36m17[] = {
    mid(11, 12), direct(12), direct(14), direct(16), direct(11), direct(13),
    direct(15), avg4(11, 12, 5, 6), mid(5, 6), avg4(0, 0, 5, 6), extrap(1, 2, 0),
    direct(5), direct(7), direct(9), direct(6), direct(8), direct(10)};
inline constexpr JointMap blaze33_to_h36m17[] = {
    mid(23, 24), direct(24), direct(26), direct(28), direct(23), direct(25),
    direct(27), avg4(23, 24, 11, 12), mid(11, 12), avg4(0, 0, 11, 12),
    extrap(2, 5, 0), direct(11), direct(13), direct(15), direct(12), direct(14),
    direct(16)};

// ============================ face: FaceMesh-468 -> dlib-68 ============================
// PeizhiYan/Mediapipe_2_Dlib_Landmarks (MIT). 5 dlib points (3,4,12,27,28) are
// the mean of two FaceMesh vertices; the rest are single vertices.
inline constexpr JointMap facemesh468_to_dlib68[] = {
    direct(127), direct(234), direct(93), mid(132, 58), mid(58, 172),
    direct(136), direct(150), direct(176), direct(152), direct(400), direct(379),
    direct(365), mid(397, 288), direct(361), direct(323), direct(454),
    direct(356),                                                        // jaw 0-16
    direct(70), direct(63), direct(105), direct(66), direct(107),       // R brow
    direct(336), direct(296), direct(334), direct(293), direct(300),    // L brow
    mid(168, 6), mid(197, 195), direct(5), direct(4),                   // nose bridge
    direct(75), direct(97), direct(2), direct(326), direct(305),        // lower nose
    direct(33), direct(160), direct(158), direct(133), direct(153), direct(144),
    direct(362), direct(385), direct(387), direct(263), direct(373), direct(380),
    direct(61), direct(39), direct(37), direct(0), direct(267), direct(269),
    direct(291), direct(321), direct(314), direct(17), direct(84), direct(91),
    direct(78), direct(82), direct(13), direct(312), direct(308), direct(317),
    direct(14), direct(87)};

// ============================ hand: MediaPipe Hands-21 -> hand-21 ============================
// COCO-WholeBody / Halpe hand order == MediaPipe order, so identity.
inline constexpr JointMap hands21_to_hand21[] = {
    direct(0),  direct(1),  direct(2),  direct(3),  direct(4),  direct(5),
    direct(6),  direct(7),  direct(8),  direct(9),  direct(10), direct(11),
    direct(12), direct(13), direct(14), direct(15), direct(16), direct(17),
    direct(18), direct(19), direct(20)};

// ============================ Source: MHR70 (InstantHMR) ============================
// SAM-3D-Body "mhr70" order (instanthmr/skeleton.py):
// 0 nose 1 Leye 2 Reye 3 Lear 4 Rear 5 Lsh 6 Rsh 7 Lel 8 Rel 9 Lhip 10 Rhip
// 11 Lknee 12 Rknee 13 Lankle 14 Rankle 15 LBigToe 16 LSmallToe 17 LHeel
// 18 RBigToe 19 RSmallToe 20 RHeel
// 21-40 right hand, 42-61 left hand: per finger (thumb,index,middle,ring,pinky)
//   base+0 tip, +1 first_joint, +2 second_joint, +3 third_joint
// 41 Rwrist 62 Lwrist (note: the wrists are NOT at COCO slots 9/10 — those are
//   the hips here)  63/64 L/R olecranon 65/66 L/R cubital fossa
//   67/68 L/R acromion 69 neck
//
// Finger order: "first_joint" is next to the TIP: first = DIP, second = PIP,
// third = MCP (thumb: first = IP, second = MCP, third = CMC). Matches
// SAM-3D-Body's mhr70 skeleton links (wrist -> third -> second -> first -> tip;
// its names are <finger>4/3/2 = tip/first/second).
//
// Neck 69 is a real joint at the base of the neck, where OpenPose puts Neck and
// Human3.6M its Thorax, so those map to direct(69) instead of mid(5,6).
// 63-68 (elbow/shoulder surface landmarks) have no counterpart in any target.
inline constexpr JointMap mhr70_to_coco17[] = {
    direct(0),  direct(1),  direct(2),  direct(3),  direct(4),  direct(5),
    direct(6),  direct(7),  direct(8),  direct(62), direct(41), direct(9),
    direct(10), direct(11), direct(12), direct(13), direct(14)};

inline constexpr JointMap mhr70_to_coco18[] = {
    direct(0),  direct(69), direct(6),  direct(8),  direct(41), direct(5),
    direct(7),  direct(62), direct(10), direct(12), direct(14), direct(9),
    direct(11), direct(13), direct(2),  direct(1),  direct(4),  direct(3)};

// The one source besides BlazePose that fills BODY_25 completely (real neck,
// both toes and the heel per foot).
inline constexpr JointMap mhr70_to_body25[] = {
    direct(0),  direct(69), direct(6),  direct(8),  direct(41), direct(5),
    direct(7),  direct(62), mid(9, 10), direct(10), direct(12), direct(14),
    direct(9),  direct(11), direct(13), direct(2),  direct(1),  direct(4),
    direct(3),  direct(15), direct(16), direct(17), direct(18), direct(19),
    direct(20)};

// Head = ear midpoint like the COCO/BlazePose tables above; Neck = real joint.
inline constexpr JointMap mhr70_to_halpe26[] = {
    direct(0),  direct(1),  direct(2),  direct(3),  direct(4),  direct(5),
    direct(6),  direct(7),  direct(8),  direct(62), direct(41), direct(9),
    direct(10), direct(11), direct(12), direct(13), direct(14), mid(3, 4),
    direct(69), mid(9, 10), direct(15), direct(18), direct(16), direct(19),
    direct(17), direct(20)};

// Same recipes as coco17_to_mpii16 with the neck joint as thorax; upper neck
// blends nose toward it (== avg4(nose, nose, shoulders) with shoulders -> neck).
inline constexpr JointMap mhr70_to_mpii16[] = {
    direct(14), direct(12), direct(10), direct(9),  direct(11), direct(13),
    mid(9, 10), direct(69), mid(0, 69), extrap(1, 2, 0), direct(41),
    direct(8),  direct(6),  direct(5),  direct(7),  direct(62)};

// Spine = midway pelvis -> thorax(neck); Neck blends nose toward the thorax.
inline constexpr JointMap mhr70_to_h36m17[] = {
    mid(9, 10), direct(10), direct(12), direct(14), direct(9), direct(11),
    direct(13), avg4(69, 69, 9, 10), direct(69), mid(0, 69), extrap(1, 2, 0),
    direct(5),  direct(7),  direct(62), direct(6), direct(8), direct(41)};

// Right hand (the first MHR hand block) in MediaPipe/COCO-WholeBody order:
// wrist, then per finger base -> tip = third, second, first, tip.
inline constexpr JointMap mhr70_to_hand21[] = {
    direct(41),                                         // WRIST
    direct(24), direct(23), direct(22), direct(21),     // thumb CMC MCP IP TIP
    direct(28), direct(27), direct(26), direct(25),     // index MCP PIP DIP TIP
    direct(32), direct(31), direct(30), direct(29),     // middle
    direct(36), direct(35), direct(34), direct(33),     // ring
    direct(40), direct(39), direct(38), direct(37)};    // pinky

inline constexpr const char* mhr70_names[] = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint",
    "right_thumb_third_joint", "right_index_tip", "right_index_first_joint",
    "right_index_second_joint", "right_index_third_joint", "right_middle_tip",
    "right_middle_first_joint", "right_middle_second_joint",
    "right_middle_third_joint", "right_ring_tip", "right_ring_first_joint",
    "right_ring_second_joint", "right_ring_third_joint", "right_pinky_tip",
    "right_pinky_first_joint", "right_pinky_second_joint",
    "right_pinky_third_joint", "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint",
    "left_thumb_third_joint", "left_index_tip", "left_index_first_joint",
    "left_index_second_joint", "left_index_third_joint", "left_middle_tip",
    "left_middle_first_joint", "left_middle_second_joint",
    "left_middle_third_joint", "left_ring_tip", "left_ring_first_joint",
    "left_ring_second_joint", "left_ring_third_joint", "left_pinky_tip",
    "left_pinky_first_joint", "left_pinky_second_joint",
    "left_pinky_third_joint", "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa",
    "right_cubital_fossa", "left_acromion", "right_acromion", "neck"};

// Native MHR70 drawing edges. Upstream SKELETON_EDGES body part (head-neck,
// torso, arms to the wrists, legs, ankle-heel) + COCO-style face edges + the
// foot (ankle-heel, then the heel/big-toe/small-toe triangle) + FULL finger
// chains wrist -> MCP(third) -> PIP(second) -> DIP(first) -> tip. 63-68 get
// no bones (dots only).
// Layout, for per-part colouring: [0, kMhr70BodyBones) body/face/feet, then
// 20 right-hand bones, then 20 left-hand bones (4 per finger, thumb first).
inline constexpr Bone mhr70_bones[] = {
    // head
    {0, 69}, {0, 1}, {0, 2}, {1, 3}, {2, 4},
    // torso
    {69, 5}, {69, 6}, {5, 9}, {6, 10}, {9, 10},
    // arms
    {5, 7}, {7, 62}, {6, 8}, {8, 41},
    // legs
    {9, 11}, {11, 13}, {10, 12}, {12, 14},
    // feet: ankle-heel + heel/big toe/small toe triangle
    {13, 17}, {17, 15}, {15, 16}, {16, 17},
    {14, 20}, {20, 18}, {18, 19}, {19, 20},
    // right hand (wrist 41, finger blocks at 21 + 4f)
    {41, 24}, {24, 23}, {23, 22}, {22, 21},
    {41, 28}, {28, 27}, {27, 26}, {26, 25},
    {41, 32}, {32, 31}, {31, 30}, {30, 29},
    {41, 36}, {36, 35}, {35, 34}, {34, 33},
    {41, 40}, {40, 39}, {39, 38}, {38, 37},
    // left hand (wrist 62, finger blocks at 42 + 4f)
    {62, 45}, {45, 44}, {44, 43}, {43, 42},
    {62, 49}, {49, 48}, {48, 47}, {47, 46},
    {62, 53}, {53, 52}, {52, 51}, {51, 50},
    {62, 57}, {57, 56}, {56, 55}, {55, 54},
    {62, 61}, {61, 60}, {60, 59}, {59, 58}};
inline constexpr int kMhr70BodyBones = 26;
inline constexpr int kMhr70HandBones = 20; // per hand
static_assert(
    std::size(mhr70_bones) == kMhr70BodyBones + 2 * kMhr70HandBones);
static_assert(std::size(mhr70_names) == 70);

// ============================ target metadata ============================
inline constexpr const char* coco17_names[] = {
    "nose",      "left_eye",  "right_eye", "left_ear",  "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle"};
inline constexpr Bone coco17_bones[] = {
    {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 6}, {5, 11}, {6, 12}, {11, 12},
    {11, 13}, {13, 15}, {12, 14}, {14, 16}, {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {0, 5}, {0, 6}};

inline constexpr const char* coco18_names[] = {
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar"};
// Edges in controlnet_aux/OpenPose `limbSeq` order, so limb i uses palette
// color i (getOpenPoseColor18) and the overlay matches an OpenPose ControlNet.
inline constexpr Bone coco18_bones[] = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10},
    {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};

inline constexpr const char* body25_names[] = {
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe",
    "RSmallToe", "RHeel"};
inline constexpr Bone body25_bones[] = {
    {1, 0}, {0, 15}, {0, 16}, {15, 17}, {16, 18}, {1, 2}, {2, 3}, {3, 4},
    {1, 5}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {10, 11}, {8, 12}, {12, 13},
    {13, 14}, {14, 19}, {19, 20}, {14, 21}, {11, 22}, {22, 23}, {11, 24}};

inline constexpr const char* halpe26_names[] = {
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow",
    "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle",
    "RAnkle", "Head", "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe",
    "RSmallToe", "LHeel", "RHeel"};
inline constexpr Bone halpe26_bones[] = {
    {0, 17}, {17, 18}, {18, 5}, {18, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {18, 19}, {19, 11}, {19, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
    {15, 24}, {15, 20}, {20, 22}, {16, 25}, {16, 21}, {21, 23}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}};

inline constexpr const char* mpii16_names[] = {
    "RAnkle", "RKnee", "RHip", "LHip", "LKnee", "LAnkle", "Pelvis", "Thorax",
    "UpperNeck", "HeadTop", "RWrist", "RElbow", "RShoulder", "LShoulder",
    "LElbow", "LWrist"};
inline constexpr Bone mpii16_bones[] = {
    {0, 1}, {1, 2}, {2, 6}, {6, 3}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {8, 9},
    {7, 12}, {12, 11}, {11, 10}, {7, 13}, {13, 14}, {14, 15}};

inline constexpr const char* h36m17_names[] = {
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot", "Spine", "Thorax",
    "Neck", "Head", "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow",
    "RWrist"};
inline constexpr Bone h36m17_bones[] = {
    {0, 1}, {1, 2}, {2, 3}, {0, 4}, {4, 5}, {5, 6}, {0, 7}, {7, 8}, {8, 9},
    {9, 10}, {8, 11}, {11, 12}, {12, 13}, {8, 14}, {14, 15}, {15, 16}};

inline constexpr const char* hand21_names[] = {
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_MCP",
    "INDEX_PIP", "INDEX_DIP", "INDEX_TIP", "MIDDLE_MCP", "MIDDLE_PIP",
    "MIDDLE_DIP", "MIDDLE_TIP", "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"};
inline constexpr Bone hand21_bones[] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9},
    {9, 10}, {10, 11}, {11, 12}, {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17},
    {0, 17}, {17, 18}, {18, 19}, {19, 20}};

// dlib-68 face contour edges (jaw, brows, nose, eyes [closed], mouth [closed]).
inline constexpr Bone dlib68_bones[] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9},
    {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16},
    {17, 18}, {18, 19}, {19, 20}, {20, 21}, {22, 23}, {23, 24}, {24, 25},
    {25, 26}, {27, 28}, {28, 29}, {29, 30}, {31, 32}, {32, 33}, {33, 34},
    {34, 35}, {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 36},
    {42, 43}, {43, 44}, {44, 45}, {45, 46}, {46, 47}, {47, 42}, {48, 49},
    {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54}, {54, 55}, {55, 56},
    {56, 57}, {57, 58}, {58, 59}, {59, 48}, {60, 61}, {61, 62}, {62, 63},
    {63, 64}, {64, 65}, {65, 66}, {66, 67}, {67, 60}};

// Mapping table for a (source, target) pair, or empty span if unsupported.
inline std::span<const JointMap> mappingFor(SourceSkeleton s, TargetSkeleton t)
{
  using S = SourceSkeleton;
  using T = TargetSkeleton;
  switch(s)
  {
    case S::Coco17:
      switch(t)
      {
        case T::OpenPoseCoco18: return coco17_to_coco18;
        case T::OpenPoseBody25: return coco17_to_body25;
        case T::Halpe26: return coco17_to_halpe26;
        case T::Mpii16: return coco17_to_mpii16;
        case T::H36m17: return coco17_to_h36m17;
        default: return {};
      }
    case S::BlazePose33:
      switch(t)
      {
        case T::Coco17: return blaze33_to_coco17;
        case T::OpenPoseCoco18: return blaze33_to_coco18;
        case T::OpenPoseBody25: return blaze33_to_body25;
        case T::Halpe26: return blaze33_to_halpe26;
        case T::Mpii16: return blaze33_to_mpii16;
        case T::H36m17: return blaze33_to_h36m17;
        default: return {};
      }
    case S::FaceMesh468:
      return t == T::Dlib68 ? std::span<const JointMap>(facemesh468_to_dlib68)
                            : std::span<const JointMap>{};
    case S::Hands21:
      return t == T::Hand21 ? std::span<const JointMap>(hands21_to_hand21)
                            : std::span<const JointMap>{};
    case S::Mhr70:
      switch(t)
      {
        case T::Coco17: return mhr70_to_coco17;
        case T::OpenPoseCoco18: return mhr70_to_coco18;
        case T::OpenPoseBody25: return mhr70_to_body25;
        case T::Halpe26: return mhr70_to_halpe26;
        case T::Mpii16: return mhr70_to_mpii16;
        case T::H36m17: return mhr70_to_h36m17;
        case T::Hand21: return mhr70_to_hand21; // right hand
        default: return {};
      }
    default: return {};
  }
}

// Skeleton edges for a target layout (for drawing / geometry output).
inline std::span<const Bone> edgesFor(TargetSkeleton t)
{
  switch(t)
  {
    case TargetSkeleton::Coco17: return coco17_bones;
    case TargetSkeleton::OpenPoseCoco18: return coco18_bones;
    case TargetSkeleton::OpenPoseBody25: return body25_bones;
    case TargetSkeleton::Halpe26: return halpe26_bones;
    case TargetSkeleton::Mpii16: return mpii16_bones;
    case TargetSkeleton::H36m17: return h36m17_bones;
    case TargetSkeleton::Dlib68: return dlib68_bones;
    case TargetSkeleton::Hand21: return hand21_bones;
    default: return {};
  }
}

// Joint names for a target layout (dlib-68 has none; geometry uses indices).
inline std::span<const char* const> namesFor(TargetSkeleton t)
{
  switch(t)
  {
    case TargetSkeleton::Coco17: return coco17_names;
    case TargetSkeleton::OpenPoseCoco18: return coco18_names;
    case TargetSkeleton::OpenPoseBody25: return body25_names;
    case TargetSkeleton::Halpe26: return halpe26_names;
    case TargetSkeleton::Mpii16: return mpii16_names;
    case TargetSkeleton::H36m17: return h36m17_names;
    case TargetSkeleton::Hand21: return hand21_names;
    default: return {};
  }
}

// Native (un-remapped) edges / names of a source layout, for drawing and
// geometry when the Skeleton control is Native. Only the layouts whose native
// table lives here: BlazePose-33 / FaceMesh / AP-10K draw from the node's own
// tables and return empty (as does dlib-68 for names).
inline std::span<const Bone> nativeEdges(SourceSkeleton s)
{
  switch(s)
  {
    case SourceSkeleton::Coco17: return coco17_bones;
    case SourceSkeleton::Hands21: return hand21_bones;
    case SourceSkeleton::Dlib68: return dlib68_bones;
    case SourceSkeleton::Mhr70: return mhr70_bones;
    default: return {};
  }
}
inline std::span<const char* const> nativeNames(SourceSkeleton s)
{
  switch(s)
  {
    case SourceSkeleton::Coco17: return coco17_names;
    case SourceSkeleton::Hands21: return hand21_names;
    case SourceSkeleton::Mhr70: return mhr70_names;
    default: return {};
  }
}

// Remap `in` (native keypoints) into `out` (target layout), reusing out's
// capacity. KP must be brace-constructible {x, y, z, confidence}. Returns false
// (out untouched) when the pair is unsupported -> caller keeps the native set.
// Synthesized joints take the weighted position and min(parent confidence);
// unmappable joints are emitted as (0,0,0) with confidence 0.
template <typename KP>
bool remap(
    SourceSkeleton s, TargetSkeleton t, std::span<const KP> in,
    std::vector<KP>& out)
{
  const auto table = mappingFor(s, t);
  if(table.empty())
    return false;

  const int n = static_cast<int>(in.size());
  out.resize(table.size());
  for(std::size_t j = 0; j < table.size(); ++j)
  {
    const JointMap& m = table[j];
    float x = 0, y = 0, z = 0, c = 0;
    bool ok = (m.n > 0);
    for(int k = 0; k < m.n; ++k)
    {
      const int i = m.idx[k];
      if(i < 0 || i >= n) // unmappable / out-of-range source
      {
        ok = false;
        break;
      }
      const KP& P = in[i];
      x += m.w[k] * P.x;
      y += m.w[k] * P.y;
      z += m.w[k] * P.z;
      // Direct joint keeps its own confidence; combinations take the min.
      c = (k == 0) ? P.confidence : std::min(c, P.confidence);
    }
    out[j] = ok ? KP{x, y, z, c} : KP{0.f, 0.f, 0.f, 0.f}; // missing -> (0,0,0)
  }
  return true;
}

} // namespace Onnx::Skel
