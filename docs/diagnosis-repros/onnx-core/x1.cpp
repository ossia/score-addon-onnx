#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <filesystem>
static std::string slurp(const char* p){ std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
int main(int argc, char** argv)
{
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "x1");
  std::string path = argv[1];
  auto bytes = slurp(argv[1]);
  auto folder = std::filesystem::path(path).parent_path().string();
  for(int mode = 0; mode < 4; mode++)
  {
    try {
      Ort::SessionOptions so;
      so.AddConfigEntry(kOrtSessionOptionsConfigUseORTModelBytesDirectly, "1"); // as create_session_options does
      if(mode == 1) so.AddConfigEntry("session.model_external_initializers_file_folder_path", folder.c_str());
      if(mode == 3) so.AddConfigEntry("session.model_external_initializers_file_folder_path", "/nonexistent");
      std::unique_ptr<Ort::Session> s;
      if(mode == 2) s = std::make_unique<Ort::Session>(env, path.c_str(), so);
      else s = std::make_unique<Ort::Session>(env, bytes.data(), bytes.size(), so);
      std::cout << "mode " << mode << ": OK, inputs=" << s->GetInputCount();
      if(argc > 2) { // run tiny model
        std::vector<float> x(4096, 1.f); int64_t sh[2]{1,4096};
        auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mi, x.data(), x.size(), sh, 2);
        const char* inn[]{"x"}; const char* outn[]{"y"};
        auto out = s->Run(Ort::RunOptions{nullptr}, inn, &in, 1, outn, 1);
        std::cout << " y0=" << out[0].GetTensorData<float>()[0];
      }
      std::cout << "\n";
    } catch(const std::exception& e) { std::cout << "mode " << mode << ": FAIL " << e.what() << "\n"; }
  }
}
