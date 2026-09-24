#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
int main(){
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR,"e2");
  for(int w : {224, 256, 128}) {
    std::vector<float> buf(3*w*w);
    std::vector<int64_t> shape{1,3,224,224};
    auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    try { auto v = Ort::Value::CreateTensor<float>(mi, buf.data(), buf.size(), shape.data(), shape.size()); std::cout << w << ": created OK\n"; }
    catch(std::exception& e){ std::cout << w << ": THROW " << e.what() << "\n"; }
  }
  // dynamic H/W shape
  std::vector<float> buf(3*224*224); std::vector<int64_t> shape{1,3,-1,-1};
  auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  try { auto v = Ort::Value::CreateTensor<float>(mi, buf.data(), buf.size(), shape.data(), shape.size()); std::cout << "dyn: created OK\n"; }
  catch(std::exception& e){ std::cout << "dyn: THROW " << e.what() << "\n"; }
}
