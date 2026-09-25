// A1 simulation: replicates AudioProcessor::operator()/runBlock/dispatchInfer
// (async path) with the real Onnx::WaveformInput/WaveformOutput. The "model" is
// identity; a job's result is applied `lat` ticks after dispatch (as the avnd
// worker_results queue does, at the start of a tick). Input is a ramp so we can
// see which input samples ever reach the output.
#include <Onnx/helpers/AudioIO.hpp>
#include <cstdio>
#include <deque>
using namespace Onnx;
struct Job { std::vector<float> in; int due; };
int main(int argc, char** argv){
  const int frames=512; const double rate=48000; const int64_t block=8192;
  for(int fixed=0; fixed<2; ++fixed)
  for(int lat : {4, 12, 20, 40}) { // inference time in ticks (1 tick = 10.7ms); block = 16 ticks
    WaveformShape ws; ws.layout=WaveLayout::BNC1; ws.channels=1; ws.block=block;
    WaveformInput in; WaveformOutput out;
    // fixed: ring sized for 4 blocks of backlog (proposed fix)
    in.prepare(ws, rate, rate, block, block, 1, fixed ? frames + 3*block : frames);
    out.prepare(1, rate, rate, block, fixed ? 3*block : frames);
    bool inprog=false; std::deque<Job> q; std::vector<float> staged, host(frames), outb(frames);
    long pushed=0, dispatched=0, dropped_blocks=0, out_nonzero=0, out_total=0, underrun_ticks=0; float ctr=1;
    const int ticks=48000*20/frames;
    for(int t=0;t<ticks;++t){
      // worker results at tick start
      while(!q.empty() && q.front().due<=t){ inprog=false; out.push(q.front().in.data(),1,(int64_t)q.front().in.size()); q.pop_front(); }
      for(auto&s:host) s=ctr++; const float* ch[1]={host.data()}; in.push(ch,1,frames); pushed+=frames;
      int guard=0;
      while(in.ready() && guard++<32){
        if(fixed && inprog) break;            // PROPOSED: don't consume while in flight
        int64_t n=in.fill(staged);
        if(inprog){ dropped_blocks++; continue; } // CURRENT: block consumed then dropped
        inprog=true; dispatched+=n; q.push_back({staged, t+lat});
      }
      float* o[1]={outb.data()}; out.pull(o,1,frames);
      int nz=0; for(auto s:outb) if(s!=0) nz++; out_nonzero+=nz; out_total+=frames; if(nz<frames && t*frames>2*block) underrun_ticks++;
    }
    printf("%s lat=%2d ticks (%.0f ms, block=%.0f ms): dispatched %.1f%% of input, dropped blocks=%ld, output non-silent %.1f%%, underrun ticks=%ld\n",
      fixed?"FIXED  ":"CURRENT", lat, lat*frames/48.0, block/48.0, 100.0*dispatched/pushed, dropped_blocks, 100.0*out_nonzero/out_total, underrun_ticks);
  }
}
