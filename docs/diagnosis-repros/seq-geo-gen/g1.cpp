#include <Onnx/helpers/GeometryIO.hpp>
#include <cstdio>
int main(){
  std::vector<std::vector<int64_t>> shapes{{1,2048,4},{1,2048,13},{1,2048,3},{1,3,3},{1,3,-1},{1,2048,50}};
  for(auto& s: shapes){ auto k=Onnx::classifyGeomOutput(s); auto pl=Onnx::detectPointLayout(s);
    std::printf("[%lld,%lld,%lld] -> %s layout=%d C=%d N=%lld\n",(long long)s[0],(long long)s[1],(long long)s[2],
      k==Onnx::GeomOutputKind::PointCloud?"PointCloud":"Data",(int)pl.layout,pl.channels,(long long)pl.count);}
}
