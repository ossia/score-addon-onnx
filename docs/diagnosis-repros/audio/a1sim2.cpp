// A1 with jitter: mean job latency below the block period, occasional spikes
// (main-thread hop stalls / CPU contention). mode 0 = current, 1 = gate fill
// while in flight (+ ring sized for backlog), 2 = gate + batch all ready blocks
// into one job (per-job overhead `hop` paid once, per-block compute `inf`).
#include <Onnx/helpers/AudioIO.hpp>
#include <cstdio>
#include <deque>
using namespace Onnx;
struct Job { std::vector<float> in; int due; };
int main(){
  const int frames=512; const double rate=48000;
  struct Case{int64_t block; int hop; int inf; int spike_every; int spike;} cases[]={
    {8192, 2, 8, 20, 14},   // 171ms block, ~107ms typical job, some 250ms spikes
    {1024, 2, 1, 1000000, 0}, // 21ms block, 3-tick job (main-thread hop 2 ticks)
    {1024, 1, 1, 30, 8}};
  for(auto c: cases){
  printf("-- block=%lld (%.0f ms), job = hop %d + %d*k ticks, spike +%d every %d jobs\n",(long long)c.block,c.block/48.0,c.hop,c.inf,c.spike,c.spike_every);
  for(int mode=0; mode<3; ++mode){
    WaveformShape ws; ws.layout=WaveLayout::BNC1; ws.channels=1; ws.block=c.block;
    WaveformInput in; WaveformOutput out;
    in.prepare(ws, rate, rate, c.block, c.block, 1, mode ? frames + 4*c.block : frames);
    out.prepare(1, rate, rate, c.block, mode ? 4*c.block : frames);
    bool inprog=false; std::deque<Job> q; std::vector<float> staged, batch, host(frames), outb(frames);
    long pushed=0, dispatched=0, lostblk=0, out_nonzero=0, out_total=0, jobs=0; float ctr=1;
    const int ticks=48000*30/frames;
    for(int t=0;t<ticks;++t){
      while(!q.empty() && q.front().due<=t){ inprog=false; out.push(q.front().in.data(),1,(int64_t)q.front().in.size()); q.pop_front(); }
      for(auto&s:host) s=ctr++; const float* ch[1]={host.data()}; in.push(ch,1,frames); pushed+=frames;
      if(mode==2){
        if(!inprog && in.ready()){ batch.clear(); int k=0; while(in.ready() && k<8){ in.fill(staged); batch.insert(batch.end(),staged.begin(),staged.end()); ++k; }
          inprog=true; dispatched+=batch.size(); jobs++; int l=c.hop+c.inf*k+((jobs%c.spike_every)==0?c.spike:0); q.push_back({batch,t+l}); }
      } else {
        int guard=0;
        while(in.ready() && guard++<32){
          if(mode==1 && inprog) break;
          int64_t n=in.fill(staged);
          if(inprog){ lostblk++; continue; }
          inprog=true; dispatched+=n; jobs++; int l=c.hop+c.inf+((jobs%c.spike_every)==0?c.spike:0); q.push_back({staged,t+l});
        }
      }
      float* o[1]={outb.data()}; out.pull(o,1,frames);
      int nz=0; for(auto s:outb) if(s!=0) nz++; out_nonzero+=nz; out_total+=frames;
    }
    printf("  %-14s dispatched %5.1f%% of input, blocks consumed-and-dropped=%ld, output non-silent %5.1f%%\n",
      mode==0?"CURRENT":mode==1?"GATE":"GATE+BATCH", 100.0*dispatched/pushed, lostblk, 100.0*out_nonzero/out_total);
  }}
}
