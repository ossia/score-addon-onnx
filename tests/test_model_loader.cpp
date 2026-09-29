// OnnxModels::ModelLoader: a node's model loads and is freed on its worker.
// The previous model runs until the new one is installed; one load is in
// flight at a time and only the latest key asked for meanwhile is loaded next;
// a model that arrives for a key no longer asked for is freed instead of
// installed; a failed load is not retried until the key changes; and a node
// destroyed while its model loads leaves nothing behind.
#include <tests/TestWorker.hpp>

#include <OnnxModels/ModelLoader.hpp>

#include <catch2/catch_test_macros.hpp>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{
struct Model
{
  std::string key;
};

struct Job
{
  OnnxModels::ModelJob<const Model, std::string> load;
};

// The models the builds made, to watch when they are freed.
std::vector<std::weak_ptr<const Model>> built;
std::vector<std::string> builtKeys;
// The model running when each load was posted.
std::vector<std::string> runningAtBuild;

// A node with the shape of the ONNX nodes: its model is one key, and a key
// starting with "bad" cannot be loaded.
struct Node
{
  struct worker
  {
    std::function<void(std::unique_ptr<Job>)> request;
    static std::function<void(Node&)> work(std::unique_ptr<Job> job)
    {
      return OnnxModels::ModelLoader<const Model, Job, std::string>::work<&Node::models, Node>(
          job->load,
          [](const std::string& key, const std::shared_ptr<const Model>& running) {
        runningAtBuild.push_back(running ? running->key : "");
        if(key.starts_with("bad"))
          throw std::runtime_error("cannot read " + key);
        auto m = std::make_shared<const Model>(key);
        built.push_back(m);
        builtKeys.push_back(key);
        return m;
      },
          [](Node& self) { self.installs++; },
          [](Node& self, std::string_view what) { self.errors.emplace_back(what); });
    }
  } worker;

  OnnxModels::ModelLoader<const Model, Job, std::string> models;
  int installs = 0;
  std::vector<std::string> errors;

  // What a node does at the start of a tick.
  void tick(const std::string& key)
  {
    if(models.requested() != key)
      models.request(worker, key);
  }

  std::string current() const { return models.model() ? models.model()->key : ""; }
};

using Kind = OnnxModels::ModelJob<const Model, std::string>::Kind;
int count(const QueuedWorker<Node, Job>& w, Kind k)
{
  int n = 0;
  for(auto& j : w.jobs)
    n += j->load.kind == k;
  return n;
}

void reset()
{
  built.clear();
  builtKeys.clear();
  runningAtBuild.clear();
}
}

TEST_CASE("ModelLoader: the previous model runs until the new one arrives", "[onnx][loader]")
{
  reset();
  Node node;
  QueuedWorker<Node, Job> w;
  w.attach(node);

  node.tick("a");
  REQUIRE(count(w, Kind::Load) == 1);
  CHECK(node.current().empty()); // nothing is built on the node's thread
  CHECK(node.models.loading());
  w.run(node);
  CHECK(node.current() == "a");
  CHECK_FALSE(node.models.loading());

  node.tick("b");
  node.tick("b");
  REQUIRE(w.jobs.size() == 1); // one request per change of key
  CHECK(node.current() == "a");

  w.run(node);
  CHECK(node.current() == "b");
  CHECK(node.installs == 2);
  CHECK(runningAtBuild == std::vector<std::string>{"", "a"});
  // "a" goes back to the worker to be freed there, not on the node's thread.
  REQUIRE(count(w, Kind::Dispose) == 1);
  CHECK_FALSE(built[0].expired());
  w.drain(node);
  CHECK(built[0].expired());
}

TEST_CASE("ModelLoader: one load at a time, the latest key next", "[onnx][loader]")
{
  reset();
  Node node;
  QueuedWorker<Node, Job> w;
  w.attach(node);

  node.tick("a");
  node.tick("b");
  node.tick("c");
  REQUIRE(w.jobs.size() == 1); // b and c wait for a
  w.run(node); // a: no longer asked for, freed; c is loaded now
  CHECK(node.current().empty());
  CHECK(node.models.loading());
  CHECK(count(w, Kind::Load) == 1);
  w.drain(node);
  CHECK(node.current() == "c");
  CHECK(node.installs == 1);
  CHECK(builtKeys == std::vector<std::string>{"a", "c"}); // b never built
  CHECK(built[0].expired());
}

TEST_CASE("ModelLoader: back to the key that is loading", "[onnx][loader]")
{
  reset();
  Node node;
  QueuedWorker<Node, Job> w;
  w.attach(node);
  node.tick("a");
  node.tick("b");
  node.tick("a");
  w.drain(node);
  CHECK(node.current() == "a");
  CHECK(builtKeys == std::vector<std::string>{"a"});
}

TEST_CASE("ModelLoader: a failed load is not retried until the key changes", "[onnx][loader]")
{
  reset();
  Node node;
  QueuedWorker<Node, Job> w;
  w.attach(node);

  node.tick("a");
  w.run(node);
  REQUIRE(node.current() == "a");

  node.tick("bad");
  w.run(node);
  CHECK(node.current().empty()); // the file that was picked cannot run
  CHECK_FALSE(node.models.loading());
  REQUIRE(node.errors.size() == 1);
  CHECK(node.errors[0] == "cannot read bad");
  CHECK(node.installs == 2);

  for(int i = 0; i < 10; i++)
    node.tick("bad");
  // Only the free of "a" was queued: no new load.
  REQUIRE(w.jobs.size() == 1);
  CHECK(count(w, Kind::Dispose) == 1);
  w.drain(node);
  CHECK(node.errors.size() == 1);

  node.tick("c");
  REQUIRE(count(w, Kind::Load) == 1);
  w.run(node);
  CHECK(node.current() == "c");
}

TEST_CASE("ModelLoader: a failure of a key no longer asked for is ignored", "[onnx][loader]")
{
  reset();
  Node node;
  QueuedWorker<Node, Job> w;
  w.attach(node);
  node.tick("bad");
  node.tick("a");
  w.drain(node);
  CHECK(node.current() == "a");
  CHECK(node.errors.empty());
}

TEST_CASE("ModelLoader: release drops the model and the load in flight", "[onnx][loader]")
{
  reset();
  Node node;
  QueuedWorker<Node, Job> w;
  w.attach(node);
  node.tick("a");
  w.run(node);
  node.tick("b");
  node.models.release(node.worker);
  CHECK(node.current().empty());
  CHECK(node.models.requested().empty());
  w.drain(node);
  CHECK(node.current().empty());
  CHECK(builtKeys == std::vector<std::string>{"a", "b"});
  for(auto& m : built)
    CHECK(m.expired());
}

// score's binding holds the node weakly: a result that completes after the
// node is gone is dropped on the worker thread, and the model with it.
TEST_CASE("ModelLoader: a node destroyed while its model loads", "[onnx][loader]")
{
  reset();
  std::mutex mut;
  std::condition_variable cv;
  bool node_gone = false;
  bool had_result = false, applied = false;
  std::thread::id worker_id, freed_on;

  struct Watched : Model
  {
    std::thread::id* freed_on{};
    ~Watched() { *freed_on = std::this_thread::get_id(); }
  };

  std::thread pool;
  {
    auto node = std::make_shared<Node>();
    node->worker.request = [&, wk = std::weak_ptr{node}](std::unique_ptr<Job> job) {
      pool = std::thread([&, wk, job = std::move(job)]() mutable {
        worker_id = std::this_thread::get_id();
        {
          std::unique_lock lock{mut};
          cv.wait(lock, [&] { return node_gone; });
        }
        auto done = OnnxModels::ModelLoader<const Model, Job, std::string>::work<&Node::models, Node>(
            job->load,
            [&](const std::string& key) {
          auto m = std::make_shared<Watched>();
          m->key = key;
          m->freed_on = &freed_on;
          built.push_back(m);
          return std::shared_ptr<const Model>(std::move(m));
        },
            [](Node&) {}, [](Node&, std::string_view) {});
        had_result = bool(done);
        if(auto n = wk.lock())
        {
          done(*n);
          applied = true;
        }
      });
    };
    node->tick("a");
  }
  {
    std::lock_guard lock{mut};
    node_gone = true;
  }
  cv.notify_one();
  pool.join();

  CHECK(had_result);
  CHECK_FALSE(applied);
  REQUIRE(built.size() == 1);
  CHECK(built[0].expired());
  CHECK(freed_on == worker_id);
}
