// Replicates the state-pairing loops of AudioAnalyzer::resolveIO (L110-136),
// AudioProcessor::resolveIO (L149-177), SequenceProcessor (L200-230) and
// VideoProcessor::pairOutput shape fallback, on real model signatures.
#include <Onnx/helpers/ModelArchetype.hpp>
#include <Onnx/helpers/AudioIO.hpp>
#include <cstdio>
using namespace Onnx;
using P = ArchIO::Port;
static const char* an(PortArchetype a){const char* n[]={"Unknown","Image","Waveform","Spectrogram","PointSet","Sequence","Vector","Latent","TokenSeq","Scalar","State"};return n[(int)a];}
static const char* nk(NodeKind k){const char* n[]={"Unknown","ImageProc","ImageGen","VideoProc","AudioProc","AudioAnalyzer","SeqProc","Geometry","TextToken"};return n[(int)k];}
static bool apSame(std::vector<int64_t> os, std::vector<int64_t> ss){ // AP/AA: dynamic in -> 1, output dyn wildcard
  bool same = os.size()==ss.size() && !os.empty();
  for(size_t k=0; same&&k<os.size(); ++k) if(os[k]>0 && os[k]!=ss[k]) same=false;
  return same;}
static void run(const char* name, ArchIO io){
  auto m = classifyModel(io);
  printf("== %s kind=%s stateful=%d\n   in:", name, nk(m.suggested), m.stateful);
  for(auto&p:m.inputs) printf(" %s=%s", p.name.c_str(), an(p.arch));
  printf("\n   out:"); for(auto&p:m.outputs) printf(" %s=%s", p.name.c_str(), an(p.arch)); printf("\n");
  // wave in (AA/AP identical)
  int wi=-1; for(int i=0;i<(int)m.inputs.size();++i){auto a=m.inputs[i].arch; if(a==PortArchetype::Waveform||a==PortArchetype::Spectrogram){wi=i;break;}}
  if(wi<0) for(int i=0;i<(int)m.inputs.size();++i){auto a=m.inputs[i].arch; if(a!=PortArchetype::RecurrentState&&a!=PortArchetype::Scalar&&a!=PortArchetype::TokenSeq){wi=i;break;}}
  if(wi<0) wi=0;
  int wo=0; for(int i=0;i<(int)m.outputs.size();++i) if(m.outputs[i].arch==PortArchetype::Waveform){wo=i;break;}
  auto ws = WaveformShape::fromInputShape(io.inputs[wi].shape);
  auto wso = WaveformShape::fromInputShape(io.outputs[wo].shape);
  auto ts = ws.tensorShape(ws.block>0?ws.block:1024);
  printf("   wave_in=%s  tensorShape=[", io.inputs[wi].name.c_str()); for(auto d:ts) printf("%lld,",(long long)d);
  printf("] block=%lld ch=%d | wave_out(AP)=%s ch=%d block=%lld\n", (long long)ws.block, ws.channels, io.outputs[wo].name.c_str(), wso.channels, (long long)wso.block);
  // AA pairing
  printf("   AA pairs:"); for(int i=0;i<(int)m.inputs.size();++i){ if(m.inputs[i].arch!=PortArchetype::RecurrentState) continue;
    auto sh=io.inputs[i].shape; for(auto&d:sh) if(d<0) d=1; int oi=-1;
    for(int o=0;o<(int)io.outputs.size();++o) if(apSame(io.outputs[o].shape, sh)){oi=o;break;}
    printf(" %s->%s", io.inputs[i].name.c_str(), oi<0?"(none)":io.outputs[oi].name.c_str());}
  printf("\n   AP pairs:"); for(int i=0;i<(int)m.inputs.size();++i){ if(m.inputs[i].arch!=PortArchetype::RecurrentState) continue;
    auto sh=io.inputs[i].shape; for(auto&d:sh) if(d<0) d=1; int oi=-1;
    for(int o=0;o<(int)io.outputs.size();++o){ if(o==wo) continue; if(apSame(io.outputs[o].shape, sh)){oi=o;break;}}
    printf(" %s->%s", io.inputs[i].name.c_str(), oi<0?"(none)":io.outputs[oi].name.c_str());}
  printf("\n   SP pairs:"); { std::vector<bool> cl(io.outputs.size());
   for(int i=0;i<(int)m.inputs.size();++i){ if(!m.stateful || m.inputs[i].arch!=PortArchetype::RecurrentState) continue; int oi=-1;
    for(int o=0;o<(int)io.outputs.size();++o){ if(cl[o]) continue; if(io.inputs[i].shape==io.outputs[o].shape){oi=o;break;}}
    if(oi>=0) cl[oi]=true; printf(" %s->%s", io.inputs[i].name.c_str(), oi<0?"(none/skipped)":io.outputs[oi].name.c_str());}}
  printf("\n");
}
int main(){
  auto F=TensorElemType::Float; auto I=TensorElemType::Int64; auto B=TensorElemType::Bool;
  run("silero-vad-v4-16k / wild silero_vad", {{{"input",{-1,-1},F},{"sr",{},I},{"h",{2,-1,64},F},{"c",{2,-1,64},F}},{{"output",{-1,1},F},{"hn",{2,-1,64},F},{"cn",{2,-1,64},F}}});
  run("silero-vad-512 (k2-fsa)", {{{"x",{1,512},F},{"h",{2,1,64},F},{"c",{2,1,64},F}},{{"prob",{-1,1},F},{"new_h",{2,-1,64},F},{"new_c",{2,-1,64},F}}});
  run("crepe-full", {{{"input",{-1,1024},F}},{{"output",{-1,360},F}}});
  run("clap fusion", {{{"longer",{1,1},B},{"mel_fusion",{1,4,1001,64},F}},{{"text_embed",{1,512},F}}});
  run("demucs htdemucs_ft_vocals", {{{"mix",{1,2,-1},F}},{{"x",{-1,-1,-1,-1},F}}});
  run("dtln1", {{{"input_2",{1,1,257},F},{"input_3",{1,2,128,2},F}},{{"activation_2",{1,1,257},F},{"tf_op_layer_stack_2",{1,2,128,2},F}}});
  run("dtln2", {{{"input_4",{1,1,512},F},{"input_5",{1,2,128,2},F}},{{"conv1d_3",{-1,1,512},F},{"tf_op_layer_stack_5",{-1,2,128,2},F}}});
  run("deep-music-enhancer", {{{"x",{-1,2,8192},F}},{{"y",{-1,2,8192},F}}});
  run("hifigan", {{{"input",{1,80,-1},F}},{{"output",{1,1,-1},F}}});
  // swapped output order (hypothetical LSTM export: cn before hn)
  run("hypothetical h,c -> cn,hn", {{{"input",{1,512},F},{"h",{2,1,64},F},{"c",{2,1,64},F}},{{"output",{1,1},F},{"cn",{2,1,64},F},{"hn",{2,1,64},F}}});
}
