#include <Onnx/helpers/TokenIO.hpp>
#include <Onnx/helpers/AudioIO.hpp>
#include <cstdio>
using namespace Onnx;
using T = TensorElemType;
struct P { std::string n; std::vector<int64_t> s; T t; };
const char* outRole(TokenOutputRole r){ switch(r){case TokenOutputRole::Waveform:return "Waveform";case TokenOutputRole::Vector:return "Vector";default:return "Tokens";} }
const char* auxRole(AuxRole r){ const char* n[]={"None","TokenIds","InputLength","Scales","NoiseScale","LengthScale","NoiseScaleW","SpeakerId","GenericFloat","GenericInt","Autoregress","Unknown"}; return n[(int)r]; }
void model(const char* name, std::vector<P> in, std::vector<P> out) {
  ArchIO io; for(auto&p:in) io.inputs.push_back({p.n,p.s,p.t}); for(auto&p:out) io.outputs.push_back({p.n,p.s,p.t});
  printf("== %s  isAutoregressive=%d isTts=%d\n", name, (int)isAutoregressive(io), (int)isTtsModel(io));
  for(auto&p:in) printf("   in  %-18s aux=%s\n", p.n.c_str(), auxRole(classifyAux(p.n,p.s,p.t)));
  for(auto&p:out) printf("   out %-18s %s\n", p.n.c_str(), outRole(classifyTokenOutput(p.n,p.s,p.t)));
}
int main(){
  model("piper amy-low", {{"input",{-1,-1},T::Int64},{"input_lengths",{-1},T::Int64},{"scales",{3},T::Float}}, {{"output",{-1,-1,1,-1},T::Float}});
  model("kitten", {{"input_ids",{1,-1},T::Int64},{"style",{1,256},T::Float},{"speed",{1},T::Float}}, {{"waveform",{-1},T::Float},{"duration",{-1},T::Int64}});
  model("kokoro", {{"tokens",{1,-1},T::Int64},{"style",{1,256},T::Float},{"speed",{1},T::Float}}, {{"audio",{-1},T::Float}});
  model("bert-nids", {{"input_ids",{-1,-1},T::Int64},{"attention_mask",{-1,-1},T::Int64}}, {{"logits",{-1,24},T::Float}});
  model("punct-en", {{"token_ids",{-1,-1},T::Int32},{"valid_ids",{-1,-1},T::Int32},{"label_lens",{-1},T::Int32}}, {{"active_case_logits",{-1,4},T::Float}});
  model("ct-transformer", {{"inputs",{-1,-1},T::Int32},{"text_lengths",{-1},T::Int32}}, {{"logits",{-1,-1,6},T::Float}});
  model("tacotron2 decoder_iter", {{"decoder_input",{1,80},T::Float},{"attention_hidden",{1,1024},T::Float},{"attention_cell",{1,1024},T::Float},{"decoder_hidden",{1,1024},T::Float},{"decoder_cell",{1,1024},T::Float},{"attention_weights",{1,-1},T::Float},{"memory",{1,-1,512},T::Float},{"mask",{1,-1},T::Bool}}, {{"decoder_output",{1,80},T::Float},{"out_attention_hidden",{1,1024},T::Float}});
  model("pocket lm_main", {{"sequence",{1,-1,32},T::Float},{"text_embeddings",{1,-1,1024},T::Float},{"state_0",{2,1,1000,16,64},T::Float},{"state_2",{1},T::Int64}}, {{"conditioning",{1,1024},T::Float},{"out_state_0",{2,1,1000,16,64},T::Float}});
  model("moonshine cached_decode", {{"args_0",{-1,-1},T::Int32},{"args_1",{-1,-1,288},T::Float},{"args_2",{1},T::Int32},{"args_3",{-1,-1,8,36},T::Float}}, {{"reversible_embedding",{-1,-1,32768},T::Float},{"functional_23",{-1,-1,8,36},T::Float}});
  model("moonshine uncached_decode", {{"args_0",{-1,-1},T::Int32},{"args_1",{-1,-1,288},T::Float},{"args_2",{1},T::Int32}}, {{"reversible_embedding",{-1,-1,32768},T::Float}});
  model("tacotron2 encoder", {{"sequences",{1,-1},T::Int64},{"sequence_lengths",{1},T::Int64}}, {{"memory",{1,-1,512},T::Float}});
  model("matcha", {{"x",{-1,-1},T::Int64},{"x_length",{-1},T::Int64},{"noise_scale",{1},T::Float},{"length_scale",{1},T::Float}}, {{"mel",{-1,80,-1},T::Float}});

  // T4 / T8: ring sizing with dynamic block -> fallback 22050
  for(double mr : {16000.0, 22050.0, 24000.0}) {
    WaveformOutput w; w.prepare(1, mr, 48000.0, 22050, 512);
    size_t cap = w.rings[0].capacity(), res = w.rs_scratch[0].capacity();
    // a 3 s utterance at model rate
    std::vector<float> utt((size_t)(3.0*mr)); for(size_t i=0;i<utt.size();++i) utt[i]=float(i);
    w.push(utt.data(), 1, (int64_t)utt.size());
    printf("model_rate=%.0f ring_cap=%zu (%.3fs@48k) rs_reserve=%zu -> after 3s push: rs_cap=%zu (REALLOC=%d) ring_size=%zu first_kept_model_sample~%.0f (%.3fs lost)\n",
      mr, cap, cap/48000.0, res, w.rs_scratch[0].capacity(), (int)(w.rs_scratch[0].capacity()!=res), w.rings[0].size(),
      w.rings[0].buf[w.rings[0].tail], w.rings[0].buf[w.rings[0].tail]/mr);
  }
  // T3: simulate operator() tick loop, job latency 40 ticks of 512 frames, utterance 1.0 s at 16k
  {
    WaveformOutput w; w.prepare(1, 16000.0, 48000.0, 22050, 512);
    std::vector<float> utt(16000); for(size_t i=0;i<utt.size();++i) utt[i]=float(i);
    bool inprog=false; int done_at=-1; int dispatches=0; std::vector<float> out(512); float* o=out.data();
    int discontinuities=0; float last=-1; int nonzero=0;
    for(int tick=0; tick<400; ++tick){
      if(inprog && tick==done_at){ inprog=false; w.push(utt.data(),1,(int64_t)utt.size()); }
      if(!inprog){ inprog=true; done_at=tick+40; dispatches++; }  // produces_audio -> always re-dispatch
      w.pull(&o,1,512);
      for(float v:out){ if(v!=0) nonzero++; if(v!=0 && last>=0 && v < last) discontinuities++; if(v!=0) last=v; }
    }
    printf("T3 sim: 400 ticks (4.27 s @48k): dispatches=%d, nonzero samples=%d, backward jumps (restarts)=%d\n", dispatches, nonzero, discontinuities);
  }
}
