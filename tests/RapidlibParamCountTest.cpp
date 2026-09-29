// The RapidLib models' "Param. count" spinbox sizes a dynamic group of knobs
// through on_controller_interaction, in a document.
#include <Process/Commands/EditPort.hpp>
#include <Process/Commands/LoadPresetCommandFactory.hpp>
#include <Process/Preset.hpp>
#include <Process/Process.hpp>

#include <Execution/BaseScenarioComponent.hpp>
#include <Execution/DocumentPlugin.hpp>
#include <Scenario/Application/Menus/ScenarioCopy.hpp>
#include <Scenario/Commands/Interval/InsertContentInInterval.hpp>
#include <Scenario/Commands/SetControllerControlValue.hpp>
#include <Scenario/Document/Interval/IntervalExecution.hpp>
#include <Scenario/Document/Interval/IntervalModel.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>

#include <core/command/CommandStack.hpp>
#include <core/document/Document.hpp>

#include <ossia/dataflow/graph_node.hpp>

#include <QApplication>
#include <QFile>

#include <score_test/App.hpp>
#include <score_test/Document.hpp>
#include <score_test/Execution.hpp>
#include <score_test/Process.hpp>
#include <score_test/Project.hpp>

#include <catch2/catch_all.hpp>

namespace
{
const QString regressor_uuid = QStringLiteral("c9613fba-6318-463c-91b0-cab4c6c7ab2b");
const QString classifier_uuid = QStringLiteral("039763f8-ea15-4900-9400-8c7f6a1c56cd");

void settle()
{
  for(int i = 0; i < 5; i++)
  {
    QCoreApplication::sendPostedEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QApplication::processEvents();
  }
}

Process::ProcessModel& add(score::Document& doc, const QString& uuid)
{
  auto proc = score::test::add_process(doc, uuid, {});
  REQUIRE(proc);
  return *proc;
}

int paramKnobs(Process::ProcessModel& p)
{
  int n = 0;
  for(auto in : p.inlets())
    if(in->name().startsWith(QStringLiteral("Param. "))
       && in->name() != QStringLiteral("Param. count"))
      n++;
  return n;
}

float knobValue(Process::ProcessModel& p, int i)
{
  return ossia::convert<float>(
      score::test::control_named(p, QStringLiteral("Param. %1").arg(i)).value());
}

void setCount(score::Document& doc, Process::ProcessModel& p, int n)
{
  auto& ctx = doc.context();
  CommandDispatcher<>{ctx.commandStack}.submit<Scenario::SetControllerControlValue>(
      score::test::control_named(p, QStringLiteral("Param. count")), ossia::value{n}, ctx);
  settle();
}

void setKnob(score::Document& doc, Process::ProcessModel& p, int i, float v)
{
  CommandDispatcher<>{doc.context().commandStack}.submit<Process::SetValue>(
      score::test::control_named(p, QStringLiteral("Param. %1").arg(i)), ossia::value{v});
  settle();
}

Process::ProcessModel& sameProcess(score::Document& doc, const Process::ProcessModel& p)
{
  auto& processes = score::test::base_interval(doc).processes;
  auto it = processes.find(p.id());
  REQUIRE(it != processes.end());
  return *it;
}

Process::ProcessModel* findProcess(score::Document& doc, const QString& uuid)
{
  const auto key = UuidKey<Process::ProcessModel>::fromString(uuid);
  for(auto& p : score::test::base_interval(doc).processes)
    if(p.concreteKey() == key)
      return &p;
  return nullptr;
}
}

TEST_CASE(
    "the regressor's parameter count adds and removes its knobs",
    "[onnx][rapidlib][dynamic]")
{
  score::test::run_in_gui_app([](const score::GUIApplicationContext& app) {
    auto doc = score::test::new_document(app);
    auto& proc = add(*doc, regressor_uuid);
    const int base = proc.inlets().size();
    CHECK(paramKnobs(proc) == 0);

    for(int n : {3, 1, 5, 0, 4})
    {
      CAPTURE(n);
      setCount(*doc, proc, n);
      CHECK(paramKnobs(proc) == n);
      CHECK(int(proc.inlets().size()) == base + n);
    }

    auto json = score::test::reload_via_json(app, *doc);
    REQUIRE(json);
    auto& reloaded = sameProcess(*json, proc);
    CHECK(paramKnobs(reloaded) == 4);

    setCount(*json, reloaded, 6);
    CHECK(paramKnobs(reloaded) == 6);
  });
}

TEST_CASE(
    "the rapidlib models' parameter count survives undo, redo and a binary reload",
    "[onnx][rapidlib][dynamic]")
{
  score::test::run_in_gui_app([](const score::GUIApplicationContext& app) {
    for(const auto& uuid : {regressor_uuid, classifier_uuid})
    {
      CAPTURE(uuid.toStdString());
      auto doc = score::test::new_document(app);
      auto& proc = add(*doc, uuid);
      const int base = proc.inlets().size();
      setCount(*doc, proc, 2);
      setCount(*doc, proc, 5);
      doc->commandStack().undo();
      settle();
      CHECK(int(proc.inlets().size()) == base + 2);
      doc->commandStack().redo();
      settle();
      CHECK(int(proc.inlets().size()) == base + 5);
      doc->commandStack().undo();
      doc->commandStack().undo();
      settle();
      CHECK(int(proc.inlets().size()) == base);
      setCount(*doc, proc, 3);
      CHECK(int(proc.inlets().size()) == base + 3);

      auto bin = score::test::reload_via_bytes(app, *doc);
      REQUIRE(bin);
      auto& reloaded = sameProcess(*bin, proc);
      CHECK(int(reloaded.inlets().size()) == base + 3);
      setCount(*bin, reloaded, 1);
      CHECK(int(reloaded.inlets().size()) == base + 1);
      setCount(*bin, reloaded, 4);
      CHECK(int(reloaded.inlets().size()) == base + 4);
    }
  });
}

// regressor-docs.score is the documentation's example, saved with one
// Regressor outlet where the spec now has two: the port count differs from the
// spec on load, so every port is rebuilt from it. The rebuild must create the
// saved knobs too, or the next count change inserts past the end.
TEST_CASE(
    "a regressor saved before the spec gained a port keeps its knobs",
    "[onnx][rapidlib][dynamic]")
{
  score::test::run_in_gui_app([](const score::GUIApplicationContext& app) {
    QFile f{QStringLiteral(SCORE_ONNX_TEST_DATA_DIR "/rapidlib/regressor-docs.score")};
    REQUIRE(f.open(QIODevice::ReadOnly));
    auto& delegates = app.interfaces<score::DocumentDelegateList>();
    auto doc = app.docManager.loadDocument(
        app, QStringLiteral("regressor-docs"), f.readAll(), JSONObject::type(),
        *delegates.begin());
    REQUIRE(doc);
    settle();
    auto proc = findProcess(*doc, regressor_uuid);
    REQUIRE(proc);

    REQUIRE(paramKnobs(*proc) == 2);
    CHECK(score::test::control_named(*proc, QStringLiteral("Param. 0")).id().val() == 17000);
    CHECK(knobValue(*proc, 0) == Catch::Approx(0.15694443881511688));
    CHECK(knobValue(*proc, 1) == Catch::Approx(0.7680555582046509));
    CHECK(proc->outlets().size() == 2);

    for(int n : {3, 1, 4})
    {
      CAPTURE(n);
      setCount(*doc, *proc, n);
      CHECK(paramKnobs(*proc) == n);
    }
    CHECK(knobValue(*proc, 0) == Catch::Approx(0.15694443881511688));
    doc->commandStack().undo();
    doc->commandStack().undo();
    doc->commandStack().undo();
    settle();
    REQUIRE(paramKnobs(*proc) == 2);
    CHECK(knobValue(*proc, 1) == Catch::Approx(0.7680555582046509));
  });
}

TEST_CASE(
    "a copied and pasted regressor keeps its knobs and their count still changes",
    "[onnx][rapidlib][dynamic]")
{
  score::test::run_in_gui_app([](const score::GUIApplicationContext& app) {
    auto doc = score::test::new_document(app);
    auto& proc = add(*doc, regressor_uuid);
    auto& ctx = doc->context();
    setCount(*doc, proc, 3);
    setKnob(*doc, proc, 2, 0.25f);

    ctx.selectionStack.pushNewSelection(Selection{&proc});
    JSONReader r;
    REQUIRE(Scenario::copySelectedProcesses(r, ctx));
    const auto copied = r.toByteArray();

    auto check = [&](score::Document& target) {
      auto& itv = score::test::base_interval(target);
      std::vector<Id<Process::ProcessModel>> before;
      for(auto& p : itv.processes)
        before.push_back(p.id());
      auto json = readJson(copied);
      CommandDispatcher<>{target.context().commandStack}.submit(
          new Scenario::Command::PasteProcessesInInterval{
              json, itv, ExpandMode::GrowShrink, QPointF{}});
      settle();
      Process::ProcessModel* pasted{};
      for(auto& p : itv.processes)
        if(!ossia::contains(before, p.id()))
          pasted = &p;
      REQUIRE(pasted);
      REQUIRE(paramKnobs(*pasted) == 3);
      CHECK(knobValue(*pasted, 2) == Catch::Approx(0.25f));
      setCount(target, *pasted, 5);
      CHECK(paramKnobs(*pasted) == 5);
      setCount(target, *pasted, 1);
      CHECK(paramKnobs(*pasted) == 1);
      target.commandStack().undo();
      target.commandStack().undo();
      settle();
      CHECK(paramKnobs(*pasted) == 3);
    };
    SECTION("in the same document") { check(*doc); }
    SECTION("in another document") { check(*score::test::new_document(app)); }
  });
}

TEST_CASE(
    "a regressor preset brings back its knob count and values",
    "[onnx][rapidlib][dynamic][preset]")
{
  score::test::run_in_gui_app([](const score::GUIApplicationContext& app) {
    auto doc = score::test::new_document(app);
    auto* proc = &add(*doc, regressor_uuid);
    auto& ctx = doc->context();
    setCount(*doc, *proc, 3);
    setKnob(*doc, *proc, 2, 0.25f);
    const auto preset = proc->savePreset();

    auto apply = [&] {
      auto& factories = app.interfaces<Process::LoadPresetCommandFactoryList>();
      auto cmd = factories.make(&Process::LoadPresetCommandFactory::make, *proc, preset, ctx);
      REQUIRE(cmd);
      CommandDispatcher<>{ctx.commandStack}.submit(cmd);
      settle();
    };

    SECTION("from fewer knobs")
    {
      setCount(*doc, *proc, 1);
      apply();
      REQUIRE(paramKnobs(*proc) == 3);
      CHECK(knobValue(*proc, 2) == Catch::Approx(0.25f));
      doc->commandStack().undo();
      settle();
      CHECK(paramKnobs(*proc) == 1);
      doc->commandStack().redo();
      settle();
      CHECK(paramKnobs(*proc) == 3);
    }
    SECTION("from more knobs")
    {
      setCount(*doc, *proc, 6);
      apply();
      CHECK(paramKnobs(*proc) == 3);
      doc->commandStack().undo();
      settle();
      CHECK(paramKnobs(*proc) == 6);
    }
    SECTION("on a new process")
    {
      proc = &add(*doc, regressor_uuid);
      apply();
      REQUIRE(paramKnobs(*proc) == 3);
      CHECK(knobValue(*proc, 2) == Catch::Approx(0.25f));
    }
  });
}

TEST_CASE(
    "the regressor's knob count changes while it is playing",
    "[onnx][rapidlib][dynamic][execution]")
{
  score::test::run_in_gui_app([](const score::GUIApplicationContext& app) {
    auto doc = score::test::new_document(app);
    auto& proc = add(*doc, regressor_uuid);
    setCount(*doc, proc, 2);

    auto& plug = doc->context().plugin<Execution::DocumentPlugin>();
    plug.reload(true, score::test::base_interval(*doc));
    score::test::run_exec(plug);
    auto node = [&] {
      REQUIRE(plug.baseScenario());
      auto& procs = plug.baseScenario()->baseInterval().processes();
      auto it = procs.find(proc.id());
      REQUIRE(it != procs.end());
      REQUIRE(it->second->node);
      return it->second->node;
    };
    const int execBase = int(node()->root_inputs().size()) - 2;

    for(int n : {5, 0, 3, 1})
    {
      CAPTURE(n);
      setCount(*doc, proc, n);
      score::test::run_exec(plug);
      score::test::run_exec(plug);
      CHECK(paramKnobs(proc) == n);
      CHECK(int(node()->root_inputs().size()) == execBase + n);
    }
    doc->commandStack().undo();
    settle();
    score::test::run_exec(plug);
    CHECK(paramKnobs(proc) == 3);
    CHECK(int(node()->root_inputs().size()) == execBase + 3);

    auto json = score::test::reload_via_json(app, *doc);
    REQUIRE(json);
    CHECK(paramKnobs(sameProcess(*json, proc)) == 3);
    plug.clear();
    settle();
  });
}
