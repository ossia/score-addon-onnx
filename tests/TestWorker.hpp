#pragma once
// score's worker protocol, run by the test: a node hands a job to
// worker.request, the thread pool runs Node::worker::work on it, and the
// returned closure is applied to the node on its own thread.
#include <OnnxModels/ModelLoader.hpp>

#include <deque>
#include <memory>
#include <utility>

// Runs one job and applies its result to the node.
template <typename Node, typename... Args>
void runJob(Node& node, Args&&... args)
{
  if(auto done = Node::worker::work(std::forward<Args>(args)...))
    done(node);
}

// A worker that runs each job right away, as score would a moment later on
// its thread pool: the node's model loads on its first tick, and its
// inference results arrive before the tick returns.
template <typename Node>
void inlineWorker(Node& node)
{
  node.worker.request = [&node](auto... args) { runJob(node, std::move(args)...); };
}

// Whether a job is a model load or free (OnnxModels::ModelLoader), rather
// than an inference.
template <typename Job>
bool isModelJob(const Job& job)
{
  return job.load.active();
}
template <typename Job>
bool isLoadJob(const Job& job)
{
  return job.load.kind == decltype(job.load)::Kind::Load;
}
template <typename Job>
bool isDisposeJob(const Job& job)
{
  return job.load.kind == decltype(job.load)::Kind::Dispose;
}

// A worker whose jobs wait in a queue until the test runs them, in any order,
// as on a busy thread pool.
template <typename Node, typename Job>
struct QueuedWorker
{
  std::deque<std::unique_ptr<Job>> jobs;

  void attach(Node& node)
  {
    node.worker.request = [this](std::unique_ptr<Job> job) { jobs.push_back(std::move(job)); };
  }

  // Runs the i-th queued job and applies its result to the node.
  void run(Node& node, std::size_t i = 0)
  {
    auto job = std::move(jobs.at(i));
    jobs.erase(jobs.begin() + i);
    runJob(node, std::move(job));
  }

  // Runs queued jobs, and the ones they queue, until none is left.
  void drain(Node& node)
  {
    while(!jobs.empty())
      run(node);
  }
};
