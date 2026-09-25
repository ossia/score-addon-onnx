#include <OnnxModels/Resnet.hpp>
#include <OnnxModels/ENet.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
static std::string slurp(const char* p){ std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
// Mimics avnd raw_file_storage::load (soundfiles.hpp): repoint bytes/filename, call update() iff the name changed.
template<typename Node, typename Port>
void load(Node& n, Port& port, const std::string& bytes, const std::string& name){
  bool changed = port.file.filename != name;
  port.file.bytes = bytes; port.file.filename = name;
  if(changed) port.update(n);
}
template<typename Node>
void tick(Node& n, std::vector<unsigned char>& img, const char* label){
  n.inputs.image.texture.bytes = img.data(); n.inputs.image.texture.width = 224; n.inputs.image.texture.height = 224; n.inputs.image.texture.changed = true;
  n();
  std::cout << label << ": invalid=" << n.inputs.model.current_model_invalid << " nout=" << n.outputs.detection.value.size();
  for(auto& d : n.outputs.detection.value) std::cout << " [" << d.name << " " << d.probability << "]";
  std::cout << "\n";
}
int main(){
  const std::string P = "/mnt/win2/models/models-presets/models/";
  std::string r18 = slurp((P+"resnet/resnet18-imagenet.onnx").c_str());
  std::string alex = slurp((P+"resnet/alexnet.onnx").c_str());
  std::string enetA = slurp((P+"emotionnet/enet_b0_8_best_afew.onnx").c_str());
  std::string fer = slurp((P+"image-processor/emotion-ferplus-8.onnx").c_str());
  std::vector<unsigned char> img(224*224*4); for(size_t i=0;i<img.size();i++) img[i]=(i*37)%251;
  std::string n1 = P+"resnet/resnet18-imagenet.onnx", n2 = P+"resnet/alexnet.onnx";
  std::string n3 = P+"emotionnet/enet_b0_8_best_afew.onnx", n4 = P+"image-processor/emotion-ferplus-8.onnx";
  {
    OnnxModels::ResnetDetector r;
    static std::string C = P+"resnet/imagenet_classes.txt"; r.inputs.classes.file.filename = C; r.inputs.classes.update(r);
    r.inputs.resolution.value = {224,224};
    load(r, r.inputs.model, r18, n1); tick(r, img, "resnet18");
    load(r, r.inputs.model, alex, n2); tick(r, img, "->alexnet (stale?)");
    OnnxModels::ResnetDetector r2;
    r2.inputs.classes.file.filename = C; r2.inputs.classes.update(r2);
    r2.inputs.resolution.value = {224,224};
    load(r2, r2.inputs.model, alex, n2); tick(r2, img, "fresh alexnet");
    // X3: resolution too small -> throw -> sticky
    r2.inputs.resolution.value = {128,128}; tick(r2, img, "alexnet res=128");
    r2.inputs.resolution.value = {224,224}; tick(r2, img, "alexnet res back to 224");
    load(r2, r2.inputs.model, alex, n2); tick(r2, img, "alexnet same file reselected");
  }
  {
    OnnxModels::EmotionNetDetector e; e.inputs.resolution.value = {224,224};
    load(e, e.inputs.model, enetA, n3); tick(e, img, "enet afew");
    load(e, e.inputs.model, fer, n4); tick(e, img, "->ferplus res=224 (stale?)");
    OnnxModels::EmotionNetDetector f; f.inputs.resolution.value = {64,64};
    load(f, f.inputs.model, fer, n4); tick(f, img, "fresh ferplus res=64");
    OnnxModels::EmotionNetDetector g; g.inputs.resolution.value = {224,224};
    load(g, g.inputs.model, fer, n4); tick(g, img, "fresh ferplus res=224");
  }
}
