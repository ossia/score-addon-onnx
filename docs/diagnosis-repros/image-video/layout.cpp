#include <Onnx/helpers/ImageModelRole.hpp>
#include <cstdio>
using namespace Onnx;
static const char* K(ImageModelKind k){const char*n[]={"Unknown","ImageToImage","ImageToMask","ImageToDepth","LatentToImage","ImageToData"};return n[(int)k];}
void show(const char* what, std::vector<int64_t> s){auto li=detail::resolveLayout(s);printf("%-40s layout=%d ch=%d w=%d h=%d\n",what,(int)li.layout,li.channels,li.w,li.h);}
int main(){
  show("hair060 in [1,4,512,512]",{1,4,512,512});
  show("hair060 out [1,512,512,2] (runtime)",{1,512,512,2});
  show("handseg out decl [-1,-1,-1,-1]",{-1,-1,-1,-1});
  show("handseg out runtime [1,2,256,256]",{1,2,256,256});
  show("pphumanseg out [1,2,-1,-1]",{1,2,-1,-1});
  show("rvm r1o [-1,16,-1,-1]",{-1,16,-1,-1});
  show("rvm pha [-1,1,-1,-1]",{-1,1,-1,-1});
  ModelIO io; io.inputs={{"input",{-1,3,-1,-1}}}; io.outputs={{"output",{-1,-1,-1,-1}}};
  printf("handseg kind=%s\n",K(classifyImage(io,0).kind));
  io.outputs={{"x",{1,2,-1,-1}}}; printf("pphumanseg kind=%s\n",K(classifyImage(io,0).kind));
  io.inputs={{"input",{1,4,512,512}}}; io.outputs={{"output",{1,512,512,2}}}; auto r=classifyImage(io,0);
  printf("hair060 kind=%s in_layout=%d in_ch=%d out_ch=%d\n",K(r.kind),(int)r.in_layout,r.in_channels,r.out_channels);
  // FILM
  io.inputs={{"time",{-1,1}},{"x0",{-1,-1,-1,3}},{"x1",{-1,-1,-1,3}}}; io.outputs={{"image",{-1,-1,-1,3}}};
  r=classifyImage(io,0); printf("film (input0=time) kind=%s latent_dim=%d\n",K(r.kind),r.latent_dim);
  // RVM outputs
  io.inputs={{"src",{-1,3,-1,-1}}}; io.outputs={{"fgr",{-1,3,-1,-1}},{"pha",{-1,1,-1,-1}},{"r1o",{-1,16,-1,-1}},{"r2o",{-1,20,-1,-1}},{"r3o",{-1,40,-1,-1}},{"r4o",{-1,64,-1,-1}}};
  for(int j=0;j<6;j++) printf("rvm out %d kind=%s\n",j,K(classifyImage(io,j).kind));
}
