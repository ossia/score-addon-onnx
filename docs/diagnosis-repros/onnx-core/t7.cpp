#include <Onnx/helpers/TokenIO.hpp>
#include <iostream>
#include <sstream>
using namespace Onnx;
int main(){
  // reads the same sig format as classify_cli
  std::string line, model; ArchIO io;
  while(std::getline(std::cin,line)){
    std::istringstream ls(line); std::string t; ls>>t;
    if(t=="MODEL"){ls>>model; io={};}
    else if(t=="IN"||t=="OUT"){std::string n,dt,d; ls>>n>>dt>>d; ArchIO::Port p; p.name=n;
      p.dtype = dt=="int64"?TensorElemType::Int64: dt=="int32"?TensorElemType::Int32: dt=="bool"?TensorElemType::Bool:TensorElemType::Float;
      if(d!="scalar"){std::stringstream ss(d); std::string x; while(std::getline(ss,x,',')) p.shape.push_back(std::stoll(x));}
      (t=="IN"?io.inputs:io.outputs).push_back(p);}
    else if(t=="END"){ auto m = classifyModel(io); std::cout<<model<<" isAutoregressive="<<isAutoregressive(io)<<" stateful="<<m.stateful<<"\n"; }
  }
}
