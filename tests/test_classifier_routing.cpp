// Model routing by the classifier (BUG-LEDGER X4), on real signatures taken
// from the model sweep. The suggested kind only labels models (classifier
// tools, MODELS.md), but the port archetypes behind it drive the nodes.
#include <Onnx/helpers/ModelArchetype.hpp>

#include <catch2/catch_test_macros.hpp>

using Onnx::NodeKind;
using Onnx::TensorElemType;

namespace
{
NodeKind kindOf(Onnx::ArchIO io)
{
  return Onnx::classifyModel(io).suggested;
}
}

TEST_CASE("Classifier routes real models to the right node", "[onnx][classify]")
{
  // CREPE: [B,1024] frames -> [B,360] pitch bins (was Sequence)
  SECTION("rvc__crepe.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"input", {-1,1024}, TensorElemType::Float}};
    io.outputs = {
        {"output", {-1,360}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::AudioAnalyzer);
  }
  // Silero: sr marks the waveform (was Sequence)
  SECTION("silero-vad__silero_vad.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"input", {-1,-1}, TensorElemType::Float},
        {"sr", {}, TensorElemType::Int64},
        {"h", {2,-1,64}, TensorElemType::Float},
        {"c", {2,-1,64}, TensorElemType::Float}};
    io.outputs = {
        {"output", {-1,1}, TensorElemType::Float},
        {"hn", {2,-1,64}, TensorElemType::Float},
        {"cn", {2,-1,64}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::AudioAnalyzer);
  }
  // SAM prompt decoder: no node hosts it (was Geometry)
  SECTION("369__vit_b_segment_anything.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"image_embeddings", {1,256,64,64}, TensorElemType::Float},
        {"point_coords", {1,-1,2}, TensorElemType::Float},
        {"point_labels", {1,-1}, TensorElemType::Float},
        {"mask_input", {1,1,256,256}, TensorElemType::Float},
        {"has_mask_input", {1}, TensorElemType::Float},
        {"orig_im_size", {2}, TensorElemType::Float}};
    io.outputs = {
        {"masks", {-1,-1,-1,-1}, TensorElemType::Float},
        {"iou_predictions", {-1,4}, TensorElemType::Float},
        {"low_res_masks", {-1,-1,-1,-1}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::Unknown);
  }
  // EdgeSAM decoder (was Geometry)
  SECTION("image_segmentation__edge_sam__edge_sam_3x_decoder.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"image_embeddings", {1,256,64,64}, TensorElemType::Float},
        {"point_coords", {1,-1,2}, TensorElemType::Float},
        {"point_labels", {1,-1}, TensorElemType::Float}};
    io.outputs = {
        {"scores", {-1,-1}, TensorElemType::Float},
        {"masks", {-1,-1,-1,-1}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::Unknown);
  }
  // moirai: bool masks and *_id metadata are not tokens (was TextToken)
  SECTION("time_series_forecasting__moirai__moirai-1.0-R-small.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"target", {-1,-1,128}, TensorElemType::Float},
        {"observed_mask", {-1,-1,128}, TensorElemType::Bool},
        {"sample_id", {-1,-1}, TensorElemType::Int64},
        {"time_id", {-1,-1}, TensorElemType::Int64},
        {"variate_id", {-1,-1}, TensorElemType::Int64},
        {"prediction_mask", {-1,-1}, TensorElemType::Bool},
        {"patch_size", {-1,-1}, TensorElemType::Int64}};
    io.outputs = {
        {"weights_logits", {-1,-1,128,4}, TensorElemType::Float},
        {"student_t_df", {-1,-1,-1}, TensorElemType::Float},
        {"student_t_loc", {-1,-1,-1}, TensorElemType::Float},
        {"student_t_scale", {-1,-1,-1}, TensorElemType::Float},
        {"normal_loc", {-1,-1,-1}, TensorElemType::Float},
        {"nb_total_count", {-1,-1,-1}, TensorElemType::Float},
        {"nb_logits", {-1,-1,-1}, TensorElemType::Float},
        {"lognormal_loc", {-1,-1,-1}, TensorElemType::Float},
        {"lognormal_scale", {-1,-1,-1}, TensorElemType::Float},
        {"loc", {-1,-1,-1}, TensorElemType::Float},
        {"scale", {-1,-1,-1}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::SequenceProcessor);
  }
  // UniAD bev [40000,1,256] is not audio (was AudioProcessor)
  SECTION("autonomous_driving__uniad__track_head.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"bev_embed", {40000,1,256}, TensorElemType::Float},
        {"query", {-1,512}, TensorElemType::Float},
        {"ref_pts", {-1,3}, TensorElemType::Float}};
    io.outputs = {
        {"output_classes", {6,-1,-1,10}, TensorElemType::Float},
        {"output_coords", {6,-1,-1,10}, TensorElemType::Float},
        {"last_ref_pts", {-1,-1,3}, TensorElemType::Float},
        {"query_feats", {6,-1,-1,256}, TensorElemType::Float},
        {"all_past_traj_preds", {6,-1,-1,-1,-1}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::SequenceProcessor);
  }
  // HWC image with no batch (was Geometry)
  SECTION("pose2__minimal-hand__detnet.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"import/prior_based_hand/Placeholder:0", {128,128,3}, TensorElemType::Uint8}};
    io.outputs = {
        {"import/prior_based_hand/strided_slice_6:0", {21,3}, TensorElemType::Float},
        {"import/prior_based_hand/strided_slice_7:0", {21,2}, TensorElemType::Int32}};
    CHECK(kindOf(io) == NodeKind::ImageProcessor);
  }
  // its bool mask was read as tokens (was TextToken)
  SECTION("tacotron2__decoder_iter.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"decoder_input", {1,80}, TensorElemType::Float},
        {"attention_hidden", {1,1024}, TensorElemType::Float},
        {"attention_cell", {1,1024}, TensorElemType::Float},
        {"decoder_hidden", {1,1024}, TensorElemType::Float},
        {"decoder_cell", {1,1024}, TensorElemType::Float},
        {"attention_weights", {1,-1}, TensorElemType::Float},
        {"attention_weights_cum", {1,-1}, TensorElemType::Float},
        {"attention_context", {1,512}, TensorElemType::Float},
        {"memory", {1,-1,512}, TensorElemType::Float},
        {"processed_memory", {1,-1,128}, TensorElemType::Float},
        {"mask", {1,-1}, TensorElemType::Bool}};
    io.outputs = {
        {"decoder_output", {1,80}, TensorElemType::Float},
        {"gate_prediction", {1,1}, TensorElemType::Float},
        {"out_attention_hidden", {1,1024}, TensorElemType::Float},
        {"out_attention_cell", {1,1024}, TensorElemType::Float},
        {"out_decoder_hidden", {1,1024}, TensorElemType::Float},
        {"out_decoder_cell", {1,1024}, TensorElemType::Float},
        {"out_attention_weights", {1,-1}, TensorElemType::Float},
        {"out_attention_weights_cum", {1,-1}, TensorElemType::Float},
        {"out_attention_context", {1,512}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::SequenceProcessor);
  }
  // a KV-cache decoder stays with TextToken, which refuses it
  SECTION("llm.int8.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"inputs_embeds", {-1,-1,1024}, TensorElemType::Float},
        {"attention_mask", {-1,-1}, TensorElemType::Int64},
        {"cache_position", {-1}, TensorElemType::Int64},
        {"cache_key_0", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_0", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_1", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_1", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_2", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_2", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_3", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_3", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_4", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_4", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_5", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_5", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_6", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_6", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_7", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_7", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_8", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_8", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_9", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_9", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_10", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_10", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_11", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_11", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_12", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_12", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_13", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_13", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_14", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_14", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_15", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_15", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_16", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_16", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_17", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_17", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_18", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_18", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_19", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_19", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_20", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_20", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_21", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_21", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_22", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_22", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_23", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_23", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_24", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_24", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_25", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_25", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_26", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_26", {-1,512,8,128}, TensorElemType::Float},
        {"cache_key_27", {-1,512,8,128}, TensorElemType::Float},
        {"cache_value_27", {-1,512,8,128}, TensorElemType::Float}};
    io.outputs = {
        {"logits", {-1,-1,151936}, TensorElemType::Float},
        {"key_delta_0", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_0", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_1", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_1", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_2", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_2", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_3", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_3", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_4", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_4", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_5", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_5", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_6", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_6", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_7", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_7", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_8", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_8", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_9", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_9", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_10", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_10", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_11", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_11", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_12", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_12", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_13", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_13", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_14", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_14", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_15", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_15", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_16", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_16", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_17", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_17", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_18", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_18", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_19", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_19", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_20", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_20", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_21", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_21", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_22", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_22", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_23", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_23", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_24", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_24", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_25", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_25", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_26", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_26", {-1,-1,8,128}, TensorElemType::Float},
        {"key_delta_27", {-1,-1,-1,128}, TensorElemType::Float},
        {"value_delta_27", {-1,-1,8,128}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::TextToken);
  }
  // a VLM decoder fed inputs_embeds stays with TextToken
  SECTION("decoder_model_merged_q4.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"inputs_embeds", {-1,-1,896}, TensorElemType::Float},
        {"attention_mask", {-1,-1}, TensorElemType::Int64},
        {"position_ids", {-1,-1}, TensorElemType::Int64},
        {"past_key_values.0.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.0.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.1.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.1.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.2.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.2.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.3.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.3.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.4.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.4.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.5.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.5.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.6.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.6.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.7.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.7.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.8.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.8.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.9.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.9.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.10.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.10.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.11.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.11.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.12.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.12.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.13.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.13.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.14.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.14.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.15.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.15.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.16.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.16.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.17.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.17.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.18.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.18.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.19.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.19.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.20.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.20.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.21.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.21.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.22.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.22.value", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.23.key", {-1,2,-1,64}, TensorElemType::Float},
        {"past_key_values.23.value", {-1,2,-1,64}, TensorElemType::Float}};
    io.outputs = {
        {"logits", {-1,-1,151646}, TensorElemType::Float},
        {"present.0.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.0.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.1.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.1.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.2.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.2.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.3.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.3.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.4.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.4.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.5.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.5.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.6.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.6.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.7.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.7.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.8.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.8.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.9.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.9.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.10.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.10.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.11.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.11.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.12.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.12.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.13.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.13.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.14.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.14.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.15.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.15.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.16.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.16.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.17.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.17.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.18.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.18.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.19.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.19.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.20.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.20.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.21.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.21.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.22.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.22.value", {-1,2,-1,64}, TensorElemType::Float},
        {"present.23.key", {-1,2,-1,64}, TensorElemType::Float},
        {"present.23.value", {-1,2,-1,64}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::TextToken);
  }
  // unchanged: Piper
  SECTION("en_US-amy-low.onnx")
  {
    Onnx::ArchIO io;
    io.inputs = {
        {"input", {-1,-1}, TensorElemType::Int64},
        {"input_lengths", {-1}, TensorElemType::Int64},
        {"scales", {3}, TensorElemType::Float}};
    io.outputs = {
        {"output", {-1,-1,1,-1}, TensorElemType::Float}};
    CHECK(kindOf(io) == NodeKind::TextToken);
  }
}
