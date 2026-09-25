#include <Onnx/helpers/Resnet.hpp>
#include <iostream>
#include <fstream>
static Ort::Value mk(std::vector<float>& v){ int64_t sh[2]{1,(int64_t)v.size()}; auto mi=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault); return Ort::Value::CreateTensor<float>(mi,v.data(),v.size(),sh,2);}
int main(int argc, char**argv){
  { std::ofstream f("/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/onnx-core/cls1000.txt"); for(int i=0;i<1000;i++) f<<"c"<<i<<"\n"; }
  OnnxModels::Resnet r; r.loadClasses("/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/onnx-core/cls1000.txt");
  Onnx::ModelSpec spec;
  std::string mode = argv[1];
  auto run=[&](std::vector<float> v){ Ort::Value t[1]{mk(v)}; std::vector<OnnxModels::Resnet::recognition_type> out; r.processOutput(spec,t,out); std::cout<<"N="<<v.size()<<":"; for(auto&o:out) std::cout<<" "<<o.name<<"="<<o.probability; std::cout<<"\n"; };
  if(mode=="shrink"){ std::vector<float> a(1000); for(int i=0;i<1000;i++) a[i]=(i==900)?20.f:0.f; run(a); std::vector<float> b(8,0.f); b[3]=5; run(b); }
  if(mode=="small"){ std::vector<float> b{1.f,2.f}; run(b); }
  if(mode=="grow"){ std::vector<float> b(8,0.f); run(b); std::vector<float> a(1000,0.f); a[900]=20; run(a); }
}
