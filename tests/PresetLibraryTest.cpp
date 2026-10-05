// The presets the ONNX packages ship (pose-detector, models-presets) against
// the current processes: each loads twice, with the process' UI built,
// through the command the preset menu submits for a wired process; each
// control it names exists, receives a value of its type that it offers, and
// each model file it names is in the library. Needs SCORE_TEST_LIBRARY_ROOT (a score
// library holding those packages); skips otherwise.
#include <Process/Dataflow/Port.hpp>
#include <Process/Dataflow/WidgetInlets.hpp>
#include <Process/DocumentPlugin.hpp>
#include <Process/Focus/FocusDispatcher.hpp>
#include <Process/Preset.hpp>
#include <Process/Process.hpp>
#include <Process/ProcessContext.hpp>
#include <Process/ProcessList.hpp>

#include <Scenario/Commands/Interval/AddOnlyProcessToInterval.hpp>
#include <Scenario/Commands/LoadPresetCommand.hpp>
#include <Scenario/Document/Interval/IntervalModel.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/serialization/JSONVisitor.hpp>

#include <core/document/Document.hpp>

#include <QApplication>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QGraphicsRectItem>
#include <QGraphicsScene>

#include <score_test/App.hpp>
#include <score_test/Document.hpp>
#include <score_test/Project.hpp>

#include <catch2/catch_all.hpp>

#include <map>

namespace
{
const char* const packages[] = {"pose-detector", "models-presets"};

void settle()
{
  for(int i = 0; i < 3; i++)
  {
    QCoreApplication::sendPostedEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QApplication::processEvents();
  }
}

QStringList presetFiles(const QString& root)
{
  QStringList files;
  for(const char* pkg : packages)
  {
    QDirIterator it{
        root + "/packages/" + pkg + "/presets", {"*.scp"}, QDir::Files,
        QDirIterator::Subdirectories};
    while(it.hasNext())
      files.push_back(it.next());
  }
  files.sort();
  return files;
}

Process::ProcessModel* addProcess(score::Document& doc, const Process::Preset& preset)
{
  auto& factories = doc.context().app.interfaces<Process::ProcessFactoryList>();
  auto* factory = factories.get(preset.key.key);
  if(!factory)
    return nullptr;
  auto& interval = score::test::base_interval(doc);
  auto* cmd = new Scenario::Command::AddOnlyProcessToInterval{
      interval, factory->concreteKey(), preset.key.effect, QPointF{}};
  CommandDispatcher<>{doc.context().commandStack}.submit(cmd);
  auto it = interval.processes.find(cmd->processId());
  return it != interval.processes.end() ? &(*it) : nullptr;
}

//! Where a file control's value points to, or empty when it names no file.
QString filePath(const ossia::value& v, const QString& root)
{
  auto str = v.target<std::string>();
  if(!str || str->empty())
    return {};
  auto path = QString::fromStdString(*str);
  if(path.startsWith("<LIBRARY>:"))
    path = root + '/' + path.mid(10);
  return path;
}

//! Values the control offers, when it offers a list; -1 when outside of it.
int indexOf(const Process::ControlInlet& ctl, const ossia::value& v)
{
  if(auto e = qobject_cast<const Process::Enum*>(&ctl))
    return e->indexOfValue(v);
  if(auto c = qobject_cast<const Process::ComboBox*>(&ctl))
    return c->indexOfValue(v);
  return 0;
}
}

TEST_CASE("The ONNX packages' presets match their processes", "[onnx][presets][model]")
{
  const QString root = qEnvironmentVariable("SCORE_TEST_LIBRARY_ROOT");
  if(root.isEmpty())
    SKIP("SCORE_TEST_LIBRARY_ROOT is not set");
  const QStringList files = presetFiles(root);
  if(files.isEmpty())
    SKIP("no pose-detector or models-presets package in " << root.toStdString());

  score::test::run_in_app([&](const score::GUIApplicationContext& ctx) {
    auto& procs = ctx.interfaces<Process::ProcessFactoryList>();
    auto& layers = ctx.interfaces<Process::LayerFactoryList>();

    int checked = 0;
    std::map<QString, int> incomplete;
    for(const QString& file : files)
    {
      const QString name = QFileInfo{file}.dir().dirName() + '/' + QFileInfo{file}.fileName();
      INFO(name.toStdString());

      QFile f{file};
      REQUIRE(f.open(QIODevice::ReadOnly));
      const QByteArray bytes = f.readAll();
      auto preset = Process::Preset::fromJson(procs, bytes);
      CHECK(preset);
      if(!preset)
        continue;

      // A fresh document each time: no preset sees the controls another set
      auto* doc = score::test::new_document(ctx);
      REQUIRE(doc);
      auto& dctx = doc->context();
      auto* proc = addProcess(*doc, *preset);
      CHECK(proc);
      if(!proc)
        continue;

      std::map<int, ossia::value> defaults;
      for(auto* inlet : proc->inlets())
        if(auto* ctl = qobject_cast<Process::ControlInlet*>(inlet))
          defaults[inlet->id().val()] = ctl->value();

      {
        Process::DataflowManager dfm;
        FocusDispatcher fd;
        Process::Context pctx{dctx, dfm, fd};
        QGraphicsScene scene;
        auto* item_root = new QGraphicsRectItem{QRectF{0., 0., 1000., 1000.}};
        scene.addItem(item_root);
        if(auto* factory = layers.findDefaultFactory(*proc))
          factory->makeItem(*proc, pctx, item_root);

        for(int i = 0; i < 2; i++)
        {
          // The command a wired process gets, as in an example: it announces
          // the ports changed, which rebuilds the UI
          CommandDispatcher<>{dctx.commandStack}.submit(
              new Scenario::Command::LoadPresetWithCablesBackup{*proc, *preset, dctx});
          settle();
        }
        scene.removeItem(item_root);
        delete item_root;
      }

      const auto json = readJson(preset->data);
      REQUIRE(json.IsArray());
      std::set<int> named;
      for(const auto& entry : json.GetArray())
      {
        const int id = entry[0].GetInt();
        const ossia::value v = JsonValue{entry[1]}.to<ossia::value>();
        named.insert(id);
        INFO("control " << id);
        Process::ControlInlet* ctl{};
        for(auto* inlet : proc->inlets())
          if(inlet->id().val() == id)
            ctl = qobject_cast<Process::ControlInlet*>(inlet);
        CHECK(ctl);
        if(!ctl)
          continue;
        INFO(ctl->name().toStdString());
        CHECK(v.get_type() == defaults[id].get_type());
        CHECK(indexOf(*ctl, v) >= 0);
        CHECK(ctl->value() == v);
        if(qobject_cast<Process::FileChooserBase*>(ctl))
        {
          const QString path = filePath(v, root);
          INFO(path.toStdString());
          CHECK((path.isEmpty() || QFileInfo{path}.isFile()));
        }
      }
      for(auto& [id, _] : defaults)
        if(!named.contains(id))
          incomplete[name]++;

      ctx.docManager.forceCloseDocument(ctx, *doc);
      settle();
      checked++;
    }

    // Not an error: such a control keeps its value when the preset loads
    for(auto& [name, n] : incomplete)
      UNSCOPED_INFO(name.toStdString() << ": " << n << " control(s) not in the preset");
    INFO(checked << " presets checked");
    CHECK(checked == files.size());
    if(!incomplete.empty())
      WARN(incomplete.size() << " presets leave some controls out");
  });
}
