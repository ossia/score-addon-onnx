#pragma once
// score's worker protocol, run by the test: a node hands a job to
// worker.request, the thread pool runs Node::worker::work on it, and the
// returned closure is applied to the node on its own thread.
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
