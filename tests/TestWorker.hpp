#pragma once
// A worker that runs each job right away, as score would a moment later on
// its thread pool: the node's model loads on its first tick, and its
// inference results arrive before the tick returns.
#include <memory>
#include <utility>

template <typename Node>
void inlineWorker(Node& node)
{
  node.worker.request = [&node](auto job) {
    if(auto done = Node::worker::work(std::move(job)))
      done(node);
  };
}
