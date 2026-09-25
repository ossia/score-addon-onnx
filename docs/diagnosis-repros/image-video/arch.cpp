#include <Onnx/helpers/ModelArchetype.hpp>
#include <cstdio>
using namespace Onnx;
int main(){
  ArchIO io;
  io.inputs={{"time",{-1,1},TensorElemType::Float},{"x0",{-1,-1,-1,3},TensorElemType::Float},{"x1",{-1,-1,-1,3},TensorElemType::Float}};
  io.outputs={{"image",{-1,-1,-1,3},TensorElemType::Float}};
  auto m=classifyModel(io);
  for(auto&p:m.inputs) printf("%s arch=%d\n",p.name.c_str(),(int)p.arch);
  printf("Image=%d Scalar=%d Latent=%d\n",(int)PortArchetype::Image,(int)PortArchetype::Scalar,(int)PortArchetype::Latent);
}
