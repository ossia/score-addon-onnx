#pragma once
// The model a node runs, loaded and freed on the node's worker.
//
// The processing thread asks for the model of a key (its file or files, and
// whatever else the model is built from) with request(); the model it runs
// meanwhile, if any, keeps running until the new one is installed at the start
// of one of its ticks. One load is in flight at a time: a key asked for while
// one loads waits for it, and only the latest key asked for meanwhile is
// loaded next, so scrolling through files does not build them all, nor several
// large models at once. A model that arrives for a key that is no longer the
// one asked for is freed, not installed. A load that fails leaves the node
// without a model and is not retried: the node asks again only when its key
// changes.
//
// The loader's jobs travel through the node's own worker, in a ModelJob
// member of the node's worker job type (`load`, or Slot for a node with
// several models): a job whose slot's kind is not None is the loader's,
// handed to ModelLoader::work first thing in the node's worker::work. The
// closures it returns reach the loader through a member pointer and hold no
// reference to the node, so a node destroyed while a model loads leaves
// nothing dangling: the binding drops the closure, and the model with it.
//
// Model may be const: the nodes whose model has no state of its own share it
// read-only with their inference jobs.
#include <OnnxModels/JobPool.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace OnnxModels
{
// A model file as a file port holds it. A file picked again, or read again
// after it changed, is a new mapping, hence a new key: loaded again, even
// under the same name.
struct ModelFile
{
  std::string path;
  const void* data{};
  std::size_t size{};

  bool operator==(const ModelFile&) const noexcept = default;

  template <typename Port>
  static ModelFile of(const Port& port)
  {
    return {std::string(port.file.filename), port.file.bytes.data(), port.file.bytes.size()};
  }
  template <typename Port>
  bool is(const Port& port) const noexcept
  {
    return data == port.file.bytes.data() && size == port.file.bytes.size()
           && path == port.file.filename;
  }
};

template <typename Model, typename Key = ModelFile>
struct ModelJob
{
  enum class Kind : uint8_t
  {
    None,
    Load,   // builds the model of `key`; `model` is the one running then
    Dispose // frees `model` on the worker
  } kind = Kind::None;
  Key key{};
  std::shared_ptr<Model> model;

  // A job of the loader, rather than of the node.
  bool active() const noexcept { return kind != Kind::None; }
};

template <
    typename Model, typename WorkerJob, typename Key = ModelFile,
    auto Slot = &WorkerJob::load>
class ModelLoader
{
public:
  using model_ptr = std::shared_ptr<Model>;
  using job_type = ModelJob<Model, Key>;

  const model_ptr& model() const noexcept { return m_model; }

  // The key of the latest request, loaded, loading or failed.
  const Key& requested() const noexcept { return m_requested; }
  // The latest request has not been answered yet.
  bool loading() const noexcept { return m_loading; }

  template <typename Worker>
  void request(Worker& worker, Key key)
  {
    m_requested = std::move(key);
    m_loading = true;
    if(!m_building)
      post(worker);
  }

  // The node has no model until the next request; one still loading is
  // dropped when it arrives.
  template <typename Worker>
  void release(Worker& worker)
  {
    m_requested = Key{};
    m_loading = false;
    dispose(worker, std::exchange(m_model, nullptr));
  }

  template <typename Worker>
  static void dispose(Worker& worker, model_ptr m)
  {
    if(!m)
      return;
    auto job = JobPool<WorkerJob>::instance().acquire();
    auto& load = (*job).*Slot;
    load.kind = job_type::Kind::Dispose;
    load.model = std::move(m);
    worker.request(std::move(job));
  }

  // Worker thread. `build(const Key&)`, or `build(const Key&, const model_ptr&
  // running)` for a model built from the one running when the load was
  // posted, returns the model or throws.
  // On the processing thread, `installed(Node&)` runs whenever model() changed
  // and `failed(Node&, std::string_view)` when the latest request failed.
  template <auto Member, typename Node, typename Build, typename Installed, typename Failed>
  static std::function<void(Node&)>
  work(job_type& slot, Build&& build, Installed installed, Failed failed)
  {
    // The pooled job goes back clean; a disposed model is freed on return.
    job_type job = std::exchange(slot, job_type{});
    if(job.kind != job_type::Kind::Load)
      return {};

    std::string error;
    try
    {
      model_ptr m = [&] {
        if constexpr(std::invocable<Build&, const Key&, const model_ptr&>)
          return build(std::as_const(job.key), std::as_const(job.model));
        else
          return build(std::as_const(job.key));
      }();
      job.model.reset();
      return [m = std::move(m), key = std::move(job.key), installed](Node& node) mutable {
        auto& self = node.*Member;
        if(!self.arrived(node.worker, key))
        {
          dispose(node.worker, std::move(m));
          return;
        }
        std::swap(self.m_model, m);
        installed(node);
        dispose(node.worker, std::move(m));
      };
    }
    catch(const std::exception& e)
    {
      error = e.what();
    }
    catch(...)
    {
      error = "unknown error";
    }

    job.model.reset();
    return [error = std::move(error), key = std::move(job.key), installed,
            failed](Node& node) mutable {
      auto& self = node.*Member;
      if(!self.arrived(node.worker, key))
        return;
      if(auto old = std::exchange(self.m_model, nullptr))
      {
        installed(node);
        dispose(node.worker, std::move(old));
      }
      failed(node, std::string_view{error});
    };
  }

private:
  template <typename Worker>
  void post(Worker& worker)
  {
    m_building = true;
    auto job = JobPool<WorkerJob>::instance().acquire();
    auto& load = (*job).*Slot;
    load.kind = job_type::Kind::Load;
    load.key = m_requested;
    load.model = m_model;
    worker.request(std::move(job));
  }

  // The load in flight is back: whether it answers the latest request. If it
  // does not, the latest key, if any, is loaded now.
  template <typename Worker>
  bool arrived(Worker& worker, const Key& key)
  {
    m_building = false;
    if(m_loading && key == m_requested)
    {
      m_loading = false;
      return true;
    }
    if(m_loading)
      post(worker);
    return false;
  }

  model_ptr m_model;
  Key m_requested{};
  bool m_loading = false;  // the latest request is not answered
  bool m_building = false; // a load is in flight
};
}
