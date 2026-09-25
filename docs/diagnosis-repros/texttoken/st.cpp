#include <Onnx/helpers/TokenIO.hpp>
#include <cstdio>
using namespace Onnx; using T=TensorElemType;
struct P{std::string n;std::vector<int64_t> s;T t;};
void st(const char* nm,std::vector<P> in,std::vector<P> out){ArchIO io;for(auto&p:in)io.inputs.push_back({p.n,p.s,p.t});for(auto&p:out)io.outputs.push_back({p.n,p.s,p.t});
 auto a=classifyModel(io); int tok=-1; for(int i=0;i<(int)a.inputs.size();++i) if(a.inputs[i].arch==PortArchetype::TokenSeq){tok=i;break;}
 printf("%-14s stateful=%d isAutoreg=%d suggested=%d tokenSeqIdx=%d\n",nm,(int)a.stateful,(int)isAutoregressive(io),(int)a.suggested,tok);}
int main(){
#include "calls.inc"
}
