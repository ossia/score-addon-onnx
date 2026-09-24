# score-addon-onnx: diagnosis of the open bugs

This is the diagnosis of every open item in `BUG-LEDGER.md`, done on 2026-09-23.
- **Out of scope:** the deprecated objects (appendix of the ledger).
- **How it was done:** nine parallel investigations. Every ledger claim was re-checked against the code; many had come from earlier agents and had never been verified.
- **Evidence:** repros used the real code where possible (the real `classifyModel` / `classify()` / `Resnet.cpp` / `ENet.cpp` compiled into harnesses, and headless `ossia-score` runs for the engine and gfx items), plus onnxruntime replicas of the node pipelines on the real models.
- **Nothing in the tree was modified** during the diagnosis.
- **Repro scripts** are in `docs/diagnosis-repros/<group>/`. Some still reference the session scratch directory for their inputs and outputs, so adjust the paths before re-running.

Status legend:
- **CONFIRMED**: reproduced, or unambiguous in the code.
- **PARTIAL**: real, but the ledger's description or example is off.
- **REFUTED**: not a bug.

## Corrections to the ledger

- **X7:** execution does honour an infinite max at build time. The bug is a resize *during playback*: `IntervalDurations` emits the stale raw `m_maxDuration`, and `IntervalExecution.cpp:114-121` forwards it to `set_max_duration`. The harness hits it because `--script` and `--autoplay` fire on the same `singleShot`. A GUI resize during playback goes through the same path. The fix is to read `maxDuration()` / `minDuration()` in the execution handlers instead of the signal payload.
- **X9:** the model state was fine all along. `Gfx::Filter::Model::loadPreset` (Filter/Process.cpp:147) never emits `programChanged()`, the only signal `Filter/Executor.cpp` listens to. So a preset loaded **while playing** leaves the running gfx node on its old shader with unregistered inlets. My earlier "GUI mode too" test also used `--autoplay`, so it hit the same race.
  - **Fix:** a one-line `programChanged();` at the end of `loadPreset`, and the same in `VSA::Model::loadPreset`.
  - **Effect:** with that, `Grid 2x2.scp` will work, and the multi-outlet test can go back to loading the preset.
- **P7:** refuted. RetinaFace and BlazeFace use the same eye order. Only the crop size differs (13–22 %).
- **P2:** the ledger's fix (fall back to 320/640) would make it worse. The PINTO wholebody exports return boxes in their internal 160×128 space, so the fallback must be 160×128.
- **R3:** ENet does *not* share the `thread_local idx` (the ledger was wrong). It is worse for Resnet, though: two Resnet nodes on one render thread corrupt each other.
- **A5:** mel vocoders don't "get raw samples". ORT throws and the node is disabled for good.
- **I5 (HAN x3):** keep the preset (None/Passthrough). The sweep's 31.2 dB for Centered/Denormalize was an artefact of an upscaled reference.
- **E2 / R5:** only a knob *smaller* than the model size throws. A larger one silently feeds a misaligned buffer.
- **L1:** no model on disk is silently mis-bound today, since LFM2 is refused at load. A mis-bind was demonstrated on a SmolLM2 with permuted outputs: it produced garbage while the smoke test still passed.

## Triage

**Small fixes with high value (S), in suggested order:**

| ID | What | Fix |
|---|---|---|
| X9 | ISF / VSA presets render nothing when loaded while playing | emit `programChanged()` in `loadPreset` |
| X8 | headless abort at end of playback | guard the `Actions::Stop` lookup with `applicationSettings.gui`, else `request_stop()`; same in RemoteControl websockets |
| X7 | a resize during playback ends at the stale max | execution handlers read `maxDuration()` / `minDuration()` |
| X2 / E1 / R1 | stale session after a model change (8 nodes) | reset `ctx` on filename change |
| X3 / R2 | sticky invalid flag | split load errors from per-frame errors, and clear on any input change |
| X1 | external-data models can't load from bytes | session option `session.model_external_initializers_file_folder_path` = model dir (verified on Llama-3.2-1B) |
| R3 | `thread_local idx` sized once | per-instance buffer, resized per call; guard N < 5 |
| AA1 (+ AudioProcessor, SequenceProcessor, VideoProcessor) | state pairing | `state_pair` computed once in `classifyModel` (name pass, then shape pass on unclaimed outputs) |
| G1 | segmentation logits routed to the point cloud | only exactly-3-channel, non-3×3 outputs count as clouds |
| S1 | Window change ignored at runtime | re-resolve when the mode changes |
| T1 | Piper rank-4 output not audio | rank 4 with `shape[2]==1` → Waveform |
| T2 | TTS rate always 22050 | `GetModelMetadata()["sample_rate"]`, then the `.onnx.json` sidecar |
| T5 | always int64 | build ints in the declared dtype |
| P1 / P3 / P4 / P6 | pose detector normalisation, anchors, class selection | see the pose section |
| F2 | noisy fp16 retry | first attempt at FATAL log level; retry with the caller's options |

**Medium changes, some shared:**
- **`ImageDecode.{hpp,cpp}`:** shared by ImageProcessor and VideoProcessor, using `decodeAll`. Fixes V1. VideoProcessor must force state outputs to Unknown in `out_kinds`.
- **`AuxInputs.hpp`:** a shared "aux input filler" that decides at load time how to fill every non-primary input, by name, dtype and archetype. Covers S2, S3, G2, IG1, AA4 and T6.
- **T3 / T4 / T8:** a TTS trigger state machine plus a one-shot player fed with a pre-resampled buffer. T1–T4 must land together for Piper.
- **A1:** gate `fill()` while a job is in flight, a bigger input ring, and batched jobs, all realtime-safe.
- **L1:** name-based binding with generic state slots. A prototype runs LFM2 end to end.
- **X10:** an "auto" inlet format that follows a float upstream. Workaround today: set the ISF inlet Format to RGBA32F.
- **X4:** nine classifier rules, tested over 205 real models, where exactly the 11 intended routings change.

**Design or large:** E4 (face crop), T9 (voice/style files), A4/A5 (spectral pipelines and mel frontend), F1 beyond FastVLM (other VLM families), P10/P11 (probe inference, generic YOLO-pose decode).

**Also found during the diagnosis** (details in the sections):
- The e4e preset lacks its mapping net.
- The DexiNed preset should use a Sigmoid mapping, which doesn't exist yet.
- FastVLM feeds the image at native size instead of 1024².
- The async workers swallow errors and retry forever.
- `create_session_with_fallback` retries with default options, dropping the chosen provider.
- The ViTPose animal models classify as human body.
- A YOLO 21-keypoint hand model is misclassified as MobileFaceNet.
- The websocket transport has the same headless crash as X8.
- AudioProcessor frees memory on the audio thread in its result lambda.
- The pose-detector package presets: four detect nothing at their current thresholds (PD3, PD10). Exact JSON edits are in the pose section; nothing was edited.

---



## Group `engine`: X7, X8

Tree: `~/ossia/score-workshop`, binary `build-developer/ossia-score`. Nothing in the tree was modified.
Repro scripts and logs are in `docs/diagnosis-repros/engine/`: `x7.js`, `ctl.js`, `run.sh`, `runG.sh`, `A.log`, `C.log`, `G.log`.
Every launch held `flock /tmp/score-harness.lock`. The environment was `--no-gui --no-restore`, `QT_QPA_PLATFORM=offscreen` and `SCORE_AUDIO_BACKEND=dummy`.

### X7: a resize during playback sends the stale raw max to the engine, although MaxInf is set
Status: CONFIRMED, but the mechanism differs from the ledger. Execution does honour `MaxInf` when it builds the interval. The bug is the *live* update path. It hits the harness because `--script` runs after `--autoplay` has already started playback: both are `QTimer::singleShot((1+wait)*1000)`, in `Engine/ApplicationPlugin.cpp:102` and `JS/ApplicationPlugin.cpp:263`.

Evidence:
- **Run A** (`x7.js` does `setIntervalDuration(root, 600 s)`, with `--autoplay --wait 1`): aborts after 25.1 s of wall time. That is startup, then play, then 15.75 s, then the X8 crash.
- **Run B** (same script, no `--autoplay`, OSC `/play T` sent at 15 s, after the script ran): still running when `timeout` hit at 60 s (rc=124). So an interval *built* after the resize plays well past 15.75 s, and `BaseScenarioComponent.cpp:100` `m_ctx.time(duration.maxDuration())` is correct.
- **Run G** (no autoplay; `/play` at 15 s; at 40 s, i.e. 25 s into playback, an OSC `/script` calls `setIntervalDuration(root, 700 s)`): aborts 3.6 s later (rc=134, `recreateBase` in the backtrace). The resize during playback made the running interval end at once, because it was already past 15.75 s. The 3-5 s delay is the backtrace symbolisation, which also shows in the control run C: a finite 5 s max ended about 9.8 s after `/play`.
- The model emits the raw value. In `Scenario/Document/Interval/IntervalDurations.cpp`, `changeAllDurations` does:
  ```
  if(!d.m_isMaxInfinite) d.m_maxDuration += delta;   // raw max left at 15.75 s
  ...
  d.maxDurationChanged(d.m_maxDuration);             // line ~240: raw, ignores MaxInf
  ```
  The same pattern appears in `setMaxDuration` (`maxDurationChanged(arg)`, line ~83) and `fixAllDurations` (`dur.maxDurationChanged(time)`, line ~186). By contrast, `setMaxInfinite` emits `maxDurationChanged(maxDuration())`, the effective value.
- The engine uses the signal's payload. In `Scenario/Document/Interval/IntervalExecution.cpp:114-121`:
  ```
  con(interval().duration, &IntervalDurations::maxDurationChanged, this, [&](TimeVal sp) {
      ... in_exec([t = ctx.time(sp), cst = m_ossia_interval] { cst->set_max_duration(t); });
  ```
  So the ossia root interval gets max = 15.75 s while it plays.

Root cause: `IntervalDurations` notifies with the raw `m_maxDuration`, which stays stale while `m_isMaxInfinite` is set. `IntervalComponent` forwards that payload to `ossia::time_interval::set_max_duration` without consulting `isMaxInfinite()`. Min has the same flaw: `changeAllDurations` emits `minDurationChanged(d.m_minDuration)` even when `m_isMinNull`.

The other suspects are cleared:
- `maxDuration()` is correct: it returns infinity when `MaxInf` is set.
- `m_ctx.time()` is the identity (`DataflowClock.cpp` `makeTimeFunction`), so infinity passes through.
- `BaseScenarioComponent` creation is correct (run B).
- The stale `m_maxDuration` in the model is harmless on its own, and it is what gets restored when the user unchecks "infinite".

GUI path: identical. Dragging the root end, or the inspector `DurationSectionWidget`, goes through `BaseScenarioIntervalResizer` → `MoveBaseEvent::updateDuration` → `changeAllDurations`. So resizing an infinite-max interval *while playing* in the GUI also truncates it. While stopped it is harmless, because the next play rebuilds from `maxDuration()`. `SetMaxDuration(…, inf=true)::redo` calls `setMaxInfinite(true)` and then `setMaxDuration(val)`. The second call re-emits the raw `val`, so even setting a max to infinite during playback leaves the engine with a finite max. Not run in the GUI; this is from reading the code.

Fix (minimal, execution side only; leaves the GUI listeners untouched): in `IntervalExecution.cpp`, ignore the payloads and read the effective values:
```cpp
con(interval().duration, &IntervalDurations::maxDurationChanged, this, [&](TimeVal) {
  if(m_ossia_interval)
    in_exec([t = ctx.time(interval().duration.maxDuration()), cst = m_ossia_interval] {
      cst->set_max_duration(t); });
});
// same for minDurationChanged with interval().duration.minDuration()
// and connect maxInfiniteChanged / minNullChanged too (setMaxInfinite already re-emits maxDurationChanged, so optional)
```
There is also a source-side alternative: emit `maxDuration()` / `minDuration()` in `setMaxDuration`, `fixAllDurations` and `changeAllDurations`. But `DurationSectionWidget::on_modelMaxDurationChanged` stores the payload into `m_max` and the spinbox, and `IntervalPresenter::on_maxDurationChanged` places the right brace from it. So that variant changes GUI behaviour and is riskier; prefer the execution-side fix.

Risk / blast radius: every executing interval (nested ones too), for live duration edits during playback only. Graphal intervals return early, so they are unaffected.

Test:
- Automated: re-run `runG.sh`. Expected: no abort within 30 s after the live resize.
- Or add an integration test: play, then `setIntervalDuration(root, …)` via `/script` while playing, then assert the engine is still running after 20 s (e.g. `/running` or a frame grab). Needs the X8 fix, or X8 masks the result as an abort.
- After the fix, drop the `setIntervalMaxDuration` workaround in `multi-outlet.js` and check that `text-render.sh` survives more than 17 s.

Effort: S

### X8: headless abort when the root interval finishes (`Actions::Stop` lookup)
Status: CONFIRMED

Evidence:
- Run A `A.log`: `terminate called after throwing an instance of 'std::out_of_range' what(): ankerl::unordered_dense::map::at(): key not found`. The backtrace goes `score::ActionManager::action<Actions::Stop>()` `ActionManager.hpp:33` ← `Execution::DocumentPlugin::recreateBase()::$_0` `DocumentPlugin.cpp:104`.
- The code:
  ```
  connect(m_base.get(), &BaseScenarioElement::finished, this, [this] {
      auto& stop_action = context().doc.app.actions.action<Actions::Stop>();
      stop_action.action()->trigger(); }, Qt::QueuedConnection);
  ```

Root cause: the transport `Actions` are registered only when there is a GUI (see the `if(ctx.applicationSettings.gui)` blocks in `ExecutionController.cpp:122ff` and `Engine/ApplicationPlugin.cpp:376`). `ActionManager::action<T>()` uses `.at()` and throws.

Fix: in `score-plugin-engine/Execution/DocumentPlugin.cpp` `recreateBase()`. `Engine/ApplicationPlugin.hpp` is already included.
```cpp
[this] {
  auto& app = context().doc.app;
  if(app.applicationSettings.gui)
    app.actions.action<Actions::Stop>().action()->trigger();
  else
    app.guiApplicationPlugin<Engine::ApplicationPlugin>().execution().request_stop();
}
```
- `request_stop()` → `m_transport->requestStop()` → the transport's stop → `trigger_stop()`. This is the same path the Stop action's `triggered` takes (queued in `ExecutionController.cpp:134`), and `trigger_stop` already runs headless when the harness sends `/stop`.
- Headless then simply stops and keeps running until `/exit`.
- `request_stop()` is enough for both modes, but keep the action with a GUI so the button state and recording stop (`Recording/ApplicationPlugin.cpp` listens on `Stop::triggered`) keep working.

Other `action<Stop/Play>()` lookups (grep of `src/`):
- Guarded by `applicationSettings.gui`:
  - `Engine/ApplicationPlugin.cpp:378,382`
  - `ExecutionController.cpp:126-138`
  - `Recording/ApplicationPlugin.cpp:54`
  - `ScenarioApplicationPlugin.cpp:437`
- GUI-only by construction: `ExecutionController.cpp:445,461` (`request_play_from_here`, `request_begin_scrub` in the `!m_clock` branch). They are connected only inside the gui block (`:141-146`).
- **Unguarded and reachable headless: `RemoteControl/Websockets/DocumentPlugin.cpp:195,199,202`** (the websocket "Play" / "Pause" / "Stop" messages). They throw the same `out_of_range` with `--no-gui`. Replace them with `execution().request_play_global(true)`, `request_play_global(false)` (pause) and `request_stop()` on `Engine::ApplicationPlugin`, or guard them as above. Not run.
- Addons `iscore-addon-network` `ClientPolicy.cpp:181` and `MasterPolicy.cpp:190` are unguarded as well. That addon is legacy and probably not built; not checked.
- Related latent issue, found by reading only: `Recording::ApplicationPlugin::record` calls `m_stopAction->trigger()`, and `m_stopAction` is null headless. Recording is GUI-driven, so this is low priority.

Risk / blast radius: only the end-of-root-interval handler, and only its headless branch changes behaviour.

Test: re-run `run.sh A 1 - 45`. Expected: no abort. With X7 unfixed, playback stops at about 15.75 s and the process stays alive until the timeout (rc=124). Every harness that sends `/exit` then exits cleanly. As a ctest: a headless run with root duration 3 s and a finite max, which asserts the process is still alive 5 s later and exits 0 on `/exit s force`.

Effort: S


## Diagnosis: group `gfx` (X9, X10)

All repros are in `diag/gfx/`. The harness is `run.sh <scene.js> <outdir> [ENV=..]`: Xvfb + llvmpipe, `--no-gui`, runs under `flock /tmp/score-harness.lock`, grabs `Window:/` 3 times over OSC, and `an.py` prints the mean RGB and the fraction of red pixels.
- Autoplay by default. With `NOAUTOPLAY=1`, `/play T` is sent after the script has finished.
- Test shader: `red.fs`, the Grid 2x2 shader with a red gap (gap 0.05), and `red-preset.json`, the same shader as a preset. When the gap renders, redfrac = 0.19.

### X9: an ISF whose program is replaced after creation renders nothing
Status: CONFIRMED, with the mechanism corrected. The model is fine. The trigger is that the preset is loaded **while the document is already playing**.
- In the harness, `--script` and `--autoplay` both fire 2 s after startup; see `JS/ApplicationPlugin.cpp` `afterStartup` and `Engine/ApplicationPlugin.cpp` `afterStartup`, both `singleShot((1 + waitAfterLoad) * 1000)`.
- In practice playback is already running when the script calls `loadPreset`.
- In GUI mode the same happens whenever you drop a preset while playing.

Evidence (all autoplay unless noted):

| scene | what | result |
|---|---|---|
| A `sA.js` | ISF from `red.fs` → Window | red gap (0.19) |
| B `sB.js` | from `red.fs` + loadPreset (identical program), no cable | red gap (0.19). Renders: same shader, output port untouched |
| C `sC.js` | from `red.fs` + loadPreset + Images → inlet 0 cable | gap only, **image missing**. Trace: no `ADDEDGE 2:0 -> 3:0` (D has it) |
| D `sD.js` | C without the loadPreset | gap + image |
| E `sE.js` | `createProcess(ISF, "")` + loadPreset | **all black**: the node still runs the Colorize shader, which has no input |
| En | E with `NOAUTOPLAY=1` (play after the script) | red gap: works |
| Cn | C with `NOAUTOPLAY=1` | gap + image: works |
| F / G | C / E, plus `g.programChanged()` from JS after loadPreset, autoplay | **both fixed** (F = image + gap, G = red gap) |

The saved `.score` of C and D differ only in their timestamps, which matches "model identical".

Root cause:
- `Gfx::Filter::Model::loadPreset` (Filter/Process.cpp:147-164) calls `setProgram`, which replaces every inlet (`clearAndDeleteLater(m_inlets)` + `setupISFModelPorts`), and then `loadFixedControls`. It never emits `programChanged()`.
- `setProgram` itself only emits `vertexChanged` / `fragmentChanged`, and only when the text differs.
- The executor's only hook is `connect(&element, &Filter::Model::programChanged, …, on_shaderChanged)` (Filter/Executor.cpp:51-53). So on a running document:
  - (a) `filter_node::set_script` never runs, and the gfx node keeps the old shader (E: Colorize → black).
  - (b) the new `Process::Inlet` objects are never registered in `SetupContext::inlets`. `LoadPresetWithCablesBackup` emits `inletsChanged`, which this executor ignores. Every cable made to the new inlets hits `inlets.find(port_snk) == end` in `SetupContext::connect_cable_impl` (ExecutionSetup.cpp:~125), and no ossia edge is created (C: input missing).
- The script-edit path (`Scenario/Commands/ScriptEditCommand.hpp` redo) explicitly calls `cmt.programChanged()`, and so do `CSF::Model::loadPreset` (CSF/Process.cpp:227) and `GeometryFilter::Model::loadPreset` (GeometryFilter/Process.cpp:217). `Filter` (ISF) and `VSA` do not.
- This is not stale port ids, `m_scriptPath`, or ProgramCache.

Fix: in `Gfx/Filter/Process.cpp` `Model::loadPreset`, emit `programChanged()` at the end, after `loadFixedControls`. Do the same in `Gfx/VSA/Process.cpp` `Model::loadPreset`, which has the same omission.
```cpp
  (void)this->setProgram(ShaderSource{type, vert, frag});
  auto controls = obj["Controls"].GetArray();
  Process::loadFixedControls(controls, *this);
  programChanged();
```
The ordering is correct inside `LoadPresetWithCablesBackup::redo`:
- `loadPreset` makes `on_shaderChanged` re-register the new inlets. It uses `unregister_node_soft(m_oldInlets…)`; the old inlets are only `deleteLater`'d, so they are still alive.
- Then `notifyAddedCables` → `on_cableCreated` finds the new inlets.

Optionally, make `setProgram` emit it itself when valid. That is riskier, because the ctor and deserialization call it.

Risk / blast radius:
- `on_shaderChanged` runs on every preset load during playback: the gfx node is re-created (`unregister_node` / `register_node`), which is the same as a script edit.
- With no execution there is no receiver, so the edit-only path is unaffected.
- Undo (`LoadPresetWithCablesBackup::undo` → `loadPreset(m_old)`) now also re-syncs the executor, which is also a fix.

Test:
- `tests/integration/multi-outlet`: switch the Grid back to `Score.loadPreset(grid, <Grid 2x2.scp>)` on a `.fs`- or `""`-created ISF. It must PASS in both `init` and `live`.
- A cheap standalone check is `diag/gfx/sE.js` + `sC.js` under autoplay: the red gap must be present and the image tile non-black.

Effort: S

### X10: float outlets are clamped to [0,1] when they feed a consumer's image input
Status: CONFIRMED. A per-inlet opt-out already exists; there is no automatic inference.

Evidence:
- The input RT is created in `RenderList::createAllInputRenderTargets` (RenderList.cpp ~321-374), with `createRenderTarget(state, spec.format, …)`.
  - The incremental edge-add path is Graph.cpp:823-855, with the same `resolveRenderTargetSpecs`. There are also re-spec paths at Graph.cpp:1021 and RenderList.cpp:1470.
  - `spec` comes from `Node::resolveRenderTargetSpecs` (Node.cpp:460), which uses only the sink node's own `renderTargetSpecs[port]`.
  - That map is filled from the model's `TextureInlet` (`setupExecution`: `exec.data.format = m_textureFormat`, TexturePort.cpp:200) through `Node::process(port, render_target_spec)`.
  - The default is `RenderTargetSpecs::format = RGBA8` (Node.hpp:59) and `TextureInlet::textureFormat = RGBA8` (TexturePort.hpp:57).
  - Nothing looks at the upstream format.
- The upstream avnd CPU node (`ImageProcessor` Depth = `halp::texture_output<"Depth", halp::r32f_texture>`) is drawn into that RT by `GenericNodeRenderer::runRenderPass` (NodeRenderer.cpp:725, a fullscreen blit). An RGBA8 target clamps it.
- Repro `sH.js` / `sI.js` (NOAUTOPLAY): ISF `two.fs` (outputs 2.0) → ISF `scale.fs` (×0.25) → Window.
  - H, default inlet format: mean **64** (= clamp(2)·0.25).
  - I, the same with `Score.inlet(g2,0).textureFormat = 8` (RGBA32F), set from JS before play: mean **128** (= 0.5). No clamp. It is saved as `"Format":8`.
  - I verified this with an ISF upstream. I did not re-run it with the avnd CPU node, but it goes through the same sink RT.

Root cause: a consumer's image-input RT format is taken only from the consumer inlet's `textureFormat`, which defaults to RGBA8. It is never inferred from the source port's texture format.

Fix (proposed, two levels):
1. **Immediate, no engine change:** set the consumer inlet's format to RGBA32F (enum value 8) or R32F (10).
   - It is in the inspector as TextureInlet "Format", and works from JS as `inlet.textureFormat = 8`.
   - The multi-outlet test can do this for `grid.bottomRight` and tighten the Depth check (r = 0.58 is currently against the *clamped* depth).
   - Limitation: ISF presets store only Controls, so `Grid 2x2.scp` cannot carry it.
2. **Automatic inference (minimal design):**
   - Add a static format hint on `score::gfx::Port` (Utils.hpp:176), e.g. `QRhiTexture::Format formatHint{QRhiTexture::UnknownFormat}`.
     - avnd `GfxNode` sets it for texture outputs from the halp texture type. `Crousti/TextureFormat.hpp` already maps halp↔QRhi formats.
     - ISF MRT outputs with a float `FORMAT` can set it too.
   - Move the spec decision into one helper (e.g. extend `Node::resolveRenderTargetSpecs`; it already takes the `RenderList&`, and the sink port's `edges` are reachable):
     - if the inlet's format is "auto", and any incoming edge's `source->formatHint` is a float format, use `RGBA32F`.
     - If `!rhi->isTextureFormatSupported(RGBA32F)` as a render target, fall back to `RGBA16F`.
     - This keeps channel semantics: R32F sampled into RGBA gives (d,0,0,1), the same as today.
   - "Auto" needs a sentinel: add `ossia::texture_format::Auto` (libossia) as the new `TextureInlet` default. Or, less clean, treat RGBA8 as auto, which silently changes existing documents: 4× memory, and values >1 are no longer clamped.
   - Also, the incremental path (Graph.cpp:823) only creates an RT when none exists. Adding a float source to a sink that already has an 8-bit RT must go through the existing `renderTargetSpecsChanged` / rt_changed surgical path (RenderList.cpp ~1454-1480), or the upgrade will not happen until the next full rebuild.

Risk / blast radius:
- Level 1: none.
- Level 2: every image-input RT allocation (full build + 2 incremental paths + the re-spec path), GPU memory for float chains, and backends without float RT support (GLES2 / WebGL1 need the fallback).
- A float RT changes blending/filtering behaviour only for float sources.

Test:
- Extend `sH` / `sI` into a ctest: float-producing ISF → scaler → Window, and assert 128 (not 64) with Auto once level 2 exists.
- In multi-outlet, assert that the Depth tile correlates with *unclamped* depth·gain.

Effort: level 1 S, level 2 M.


## Diagnosis: group `onnx-core` (X1–X5, E1–E6, R1–R5)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx` (A). Nothing in the tree was modified.
The helper and node sources are byte-identical to score-master's addon: `cmp` was run on ModelArchetype, OnnxContext, Resnet.hpp, Resnet.cpp, ENet.*, Utils, Images and TokenIO.
The classifier tests (`tests/classify_cli.cpp`, `classify_real_models.py`) exist **only** in score-master's addon, so they were built from there.
Scratch dir: `S=docs/diagnosis-repros/onnx-core/`.
ORT used for the repros: the build's own `build-developer/_deps/onnxruntime-src` (1.27.1).

Repro artefacts in `$S`:
- `x1/x1.cpp`: external data.
- `x2.cpp`: drives the real `ResnetDetector` / `EmotionNetDetector` (Resnet.cpp + ENet.cpp compiled in) through the same bytes/filename/`update()` sequence as avnd's `raw_file_storage::load`.
- `r3.cpp`: `Resnet::processOutput`.
- `e2.cpp`: `CreateTensor` size checks.
- `t7.cpp`: `isAutoregressive`.
- `x4/Onnx/helpers/ModelArchetype.hpp`: the patched classifier with the proposed X4 rules.
- `sigs_all.txt`: 205 real signatures from wild, wild2 and pinto.
- `base.txt` / `new.txt`: classifier output before and after the patch.
- `check_expected.py`: the EXPECTED table check.

---

### X1 — Nodes cannot load models with external data (`.onnx_data`)
Status: CONFIRMED, and the fix is verified.
Evidence: `OnnxContext.hpp:296-299` is `explicit OnnxRunContext(std::string_view bytes)` followed by `session(env, bytes.data(), bytes.size(), session_options)`. There is no path, so ORT resolves the external file against the **process CWD**.
`$S/x1/x1 <model>` tries four modes (mode 0 is the current code):
```
tiny.onnx (+tiny.onnx_data, made with onnx.save(save_as_external_data)):
mode 0 (bytes, as today):      FAIL External data path for model loaded from bytes escapes working directory ... "/tiny.onnx_data" does not exist
mode 1 (bytes + folder option): OK, y0=33546.2
mode 2 (session from path):     OK, y0=33546.2
mode 3 (bytes + wrong folder):  FAIL ... "/nonexistent/tiny.onnx_data"
Llama-3.2-1B-Instruct/onnx/model_q4.onnx (+1.7 GB model_q4.onnx_data):
mode 0 FAIL (same error); mode 1 OK inputs=34; mode 2 OK inputs=34
```
Root cause: a session built from a buffer has no base directory. ORT 1.27 supports `kOrtSessionOptionsModelExternalInitializersFileFolderPath` (`"session.model_external_initializers_file_folder_path"`, `onnxruntime_session_options_config_keys.h:361`) for exactly this case, and it works together with the `UseORTModelBytesDirectly=1` entry the add-on sets. Loading happens to work today only when score's CWD is the model folder.
Fix: give `OnnxRunContext` an optional `std::string_view model_path` and set the entry on the options before building the session:
```cpp
explicit OnnxRunContext(std::string_view bytes, std::string_view model_path = {})
  : env(make_env("ossia"))
  , session_options(withModelFolder(create_session_options(opts), model_path))
  , session(env, bytes.data(), bytes.size(), session_options) {...}
static Ort::SessionOptions withModelFolder(Ort::SessionOptions so, std::string_view p) {
  if(!p.empty()) so.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath,
                                   std::filesystem::path(p).parent_path().string().c_str());
  return so; }
```
Then pass `inputs.model.file.filename` at every `make_*<Onnx::OnnxRunContext>(…bytes)` call:
- Resnet.cpp:29, ENet.cpp:40
- TextToken.cpp:110, VideoProcessor.cpp:551, SequenceProcessor.cpp:162, GeometryProcessor.cpp:248, AudioProcessor.cpp:93, ImageGenerator.cpp:230/239, ImageProcessor.cpp:471, AudioAnalyzer.cpp:73
- PoseDetector.cpp:248/254/261
- the legacy nodes (TRTPose:36, BlazePose:32, YOLO-*, DepthAnything2:38)

This beats loading from the path, because the mmapped bytes are already in hand.
Risk / blast radius: every node's session creation. The setting is ignored for models without external data. `create_session_with_fallback` is path-based and unaffected. `readModelSpec(string_view)` (OnnxContext.hpp:409) and `GAN.cpp:524/573/603` also pass bytes; that is the legacy GAN path.
Test: add `x1.cpp`'s tiny model (generated with 20 lines of Python) as a unit test that builds `OnnxRunContext(bytes, path)` from a different CWD and runs it. Then update MODELS.md "Read this first" item 2.
Effort: S

### X2 — Stale session after a model change (`if(!ctx)` pattern)
Status: CONFIRMED (reproduced with the real node code).
Evidence:
- `Utils.hpp:154-157`: `ModelPort::update(OnnxObject&)` only does `current_model_invalid = bytes.size() < 32`. It cannot reach the node's `ctx`.
- avendish `binding/ossia/soundfiles.hpp:398-409` (the GPU-state overload, which these texture nodes use via `GpuUtils.hpp:97`) repoints `port.file.bytes` and `filename`, then calls `port.update(state)` only `if(changed)` on the filename.
- `Resnet.cpp:26` / `ENet.cpp:37`: `if (!this->ctx) ctx = make_unique<OnnxRunContext>(bytes)` never re-runs.

`$S/x2` output:
```
resnet18:            [nematode 0.0302] [hook 0.0274] ...
->alexnet (stale?):  [nematode 0.0302] [hook 0.0274] ...     <- identical: still resnet18's session
fresh alexnet:       [window screen 0.515] [lacewing 0.0445] ...
enet afew:           [Anger 0.1375] ... [Neutral 0.3358] ...
->ferplus (stale?):  [Anger 0.1375] ... [Neutral 0.3358] ... <- identical: still afew's session
```
Nodes with the bare `if(!ctx)` pattern (all stale on a model change):
- `Resnet.cpp:26`, `ENet.cpp:37`
- legacy: `BlazePose.cpp:30`, `DepthAnything2.cpp:35`, `TRTPose.cpp:34`, `YOLO-pose.cpp:33`, `YOLO-blob.cpp:40`, `YOLO-segmentation.cpp:37`

Nodes that already handle it:
- `!ctx || lastModelPath != filename`: AudioProcessor:215, AudioAnalyzer:166, GeometryProcessor:268, SequenceProcessor:308, ImageProcessor:530, VideoProcessor:571, TextToken:275, ImageGenerator:270
- PoseDetector.cpp:205-221 (`m_last_model` etc.)
- GenerativeImageGAN / ImageToImageGAN (`needsReinitialization` compares filenames)
- FastVLM.cpp:14, QwenLLM.cpp:17

Root cause: session creation is keyed only on "no ctx yet". The port-change hook exists but receives `OnnxObject&` and has no access to `ctx`.
Fix (minimal, the same as the new nodes): in Resnet/ENet, add `std::string lastModelPath;` and write
```cpp
if(!ctx || lastModelPath != inputs.model.file.filename) {
  ctx = std::make_unique<Onnx::OnnxRunContext>(inputs.model.file.bytes /*, filename for X1*/);
  lastModelPath = inputs.model.file.filename; }
```
Apply the same to the six legacy nodes if they are kept.
Risk: none beyond these nodes.
A side note: with the stale ctx, a `.ort`-format model (built with `UseORTModelBytesDirectly=1`) would keep pointing at the old mmap, which `raw_file_storage::load` releases, so it would dangle. `.onnx` protobufs are copied, so this is theoretical given the `*.onnx` filter.
Test: `$S/x2.cpp` is the test. Turn it into a unit test that asserts the output after switching to alexnet equals the output of a fresh alexnet node.
Effort: S

### X3 — `current_model_invalid` is sticky
Status: CONFIRMED, and worse than described: even re-selecting the same file does not clear it.
Evidence:
- `Resnet.cpp:63-66` / `ENet.cpp:74-77`: `catch(...) { current_model_invalid = true; }` wraps session creation **and** every per-frame step.
- The only reset is `ModelPort::update`, which avnd calls only when the *filename changes* (soundfiles.hpp:400, `if(changed)`).
- `$S/x2`:
```
fresh alexnet:                 invalid=0 [window screen 0.515] ...
alexnet res=128:               invalid=1 [window screen 0.515] ...   (throws: see E2)
alexnet res back to 224:       invalid=1 [window screen 0.515] ...   <- stays dead
alexnet same file reselected:  invalid=1 [window screen 0.515] ...   <- stays dead
```
- The last detections also stay on the outlet forever: `outputs.detection.value.clear()` comes after the throw point.
- The same catch-all pattern is in every node that sets the flag: ImageProcessor.cpp:548, VideoProcessor.cpp:591, SequenceProcessor.cpp:340, GeometryProcessor.cpp:282, AudioProcessor.cpp:244, AudioAnalyzer.cpp:187, TextToken.cpp:323, ImageGenerator.cpp:283, and all the legacy nodes.

Root cause: load failures and per-frame failures share one permanent flag, and nothing except a filename change clears it.
Fix: split the two. Only a failed `OnnxRunContext` construction sets `current_model_invalid`, via its own try/catch around the ctx creation. The outer catch clears the outputs and returns, logging once:
```cpp
if(!ctx || lastModelPath != fn) {
  try { ctx = ...; lastModelPath = fn; }
  catch(...) { inputs.model.current_model_invalid = true; return; } }
... catch(...) { outputs.detection.value.clear(); }   // per-frame: skip the frame only
```
Optionally, also clear the flag from the resolution port's `update` (or when any control changes).
Risk: a model that throws on every frame now throws, and gets caught, on every frame. That costs CPU, not correctness. Rate-limit the stderr print.
Test: extend `x2.cpp`: after res=128 then res=224, expect `invalid=0` and a fresh detection.
Effort: S (Resnet/ENet); M if applied to all the new nodes.

### X4 — Classifier misroutes
Status: CONFIRMED for every listed model (run through the real `classifyModel` via `classify_cli`).
Evidence (baseline, `$S/base.txt`):
```
rvc__crepe                     kind=SequenceProcessor  in=Latent                         ([-1,1024] "input")
silero-vad                     kind=SequenceProcessor  in=Latent,Scalar,State,State      (input [-1,-1], sr int64)
edge_sam_3x_decoder            kind=GeometryProcessor  in=Unknown,PointSet,Latent        (point_coords [1,-1,2] → PointSet)
369__vit_{b,l,h}_segment_anything kind=GeometryProcessor
moirai-1.0-R-{small,base}      kind=TextToken          in=Sequence,Sequence(bool mask),TokenSeq×4(sample_id,time_id,variate_id,prediction_mask bool),...
uniad__track_head              kind=AudioProcessor     bev_embed [40000,1,256] → Waveform ([.,1,N] rule ignores s[0])
minimal-hand detnet            kind=GeometryProcessor  uint8 [128,128,3] → PointSet (s[2]==3)
```
Two extra findings:
- `bool` tensors count as token ids, because `isIntType` includes Bool. Tacotron's `mask`, moirai's `prediction_mask` and dab-detr's mask are all read as TokenSeq.
- `hasIn(TokenSeq)` routes to TextToken even when the tokens are an auxiliary input.

Root cause: `classifyPort` looks at each port in isolation, and `classifyModel` picks a kind from "any input has X" instead of from the primary input.
Fix: concrete rules, implemented and tested in `$S/x4/Onnx/helpers/ModelArchetype.hpp` (diff it against A's copy).
- a. `classifyPort`: `dtype == Bool` → `Unknown` (a mask, never tokens).
- b. `classifyPort`: an input of rank ≥ 2 with an integer dtype whose name ends in `_id` (singular) → `Unknown` (index metadata; `*_ids` is still tokens).
- c. rank 3: `!pointName && s[2]∈{3,4} && s[0]>=16 && s[1]>=16` → `Image` (HWC without batch), checked before the PointSet rule.
- d. rank 3 Waveform `[.,1|2,N]` additionally requires `s[0] <= 16` (a batch-like leading dim).
- e. `classifyModel`: if an input is named `sr`/`sample_rate`/`sampling_rate`, **or** the CREPE fingerprint holds (single input `[.,1024]` → single output `[.,360]`), retag the first Latent/Sequence input as `Waveform`.
- f. `classifyModel`: inputs `image_embeddings` + `point_coords`/`point_labels` → `suggested = Unknown` (a SAM prompt decoder; no node hosts the encoder→decoder chain).
- g. Route to TextToken only when the **primary** input is TokenSeq. The primary input is the first input that is not Scalar/RecurrentState/Unknown.
- h. rank-4 NHWC `Image` requires H and W both concrete or both symbolic.
- i. The rank-4 symbolic-channel fallback `s[1]<=0 → Image` additionally requires `s[0] <= 1` and `s[2] <= 0`. This is needed so moirai's `weights_logits [?,?,128,4]` and UniAD's `[6,?,?,10]` stop being Images; without it, (b) and (g) turn moirai into an ImageGenerator.

Result over 205 real models (`diff base.txt new.txt`): exactly **11 kind changes**, and the pose-classifier column is unchanged:
```
crepe → AudioAnalyzer; silero → AudioAnalyzer (stateful kept); SAM b/l/h + EdgeSAM → Unknown;
moirai small/base → SequenceProcessor; uniad track_head → SequenceProcessor; detnet → ImageProcessor;
tacotron2__decoder_iter → SequenceProcessor (was TextToken)
```
Port-level side effects, all improvements: fairface's int `*_id` outputs stay Scalar (rule b is inputs-only); CLAP's `longer` bool and dab-detr's bool mask become Unknown; e2pose's `[?,?,17,3]` outputs are no longer Image.
EXPECTED table (`check_expected.py`): baseline 65/65 pass; new 64/65. The one failure is `tacotron2__decoder_iter` expected `("TextToken", True)`.
That expectation only held because its bool `mask` was read as tokens. Update it to `("SequenceProcessor", True)` together with T7 (below), since TextToken never refused it anyway.
Also fix `DEFAULT_DIRS` in `classify_real_models.py`, which points to `/mnt/models/...`; the models are in `/mnt/sdd1/models/...`.
Risk: kind is advisory, but it drives port roles.
- (a)/(b)/(g) change TextToken's token-input choice only for models where tokens were not the primary input.
- (c) could catch a real point set `[N≥16, M≥16, 3]` with no batch dim. None exists in the corpus.
- (e)'s CREPE rule is a fingerprint. A generic "D=1024 means waveform" rule would misroute 1024-d embedding heads, so it was not used.
- UniAD is still not really hostable; SequenceProcessor is just the least-wrong generic.

Test: add the 11 models to `EXPECTED` in `tests/classify_real_models.py` with the new kinds (crepe/silero → AudioAnalyzer, SAM ×4 → Unknown, moirai ×2 / uniad → SequenceProcessor, detnet → ImageProcessor).
Effort: M

### X5 — MODELS.md claims tacotron2 decoder_iter is refused
Status: CONFIRMED (T7 verified).
Evidence: `$S/t7` runs A's `Onnx::isAutoregressive` on the real signatures:
```
tacotron2__decoder_iter.onnx      isAutoregressive=0 stateful=1
cached_decode.int8.onnx (moonshine) isAutoregressive=0
lm_main.int8.onnx (pocket-tts)    isAutoregressive=0 stateful=1
blip2-opt-2.7b.onnx               isAutoregressive=1   (control: KV names are caught)
```
- `TokenIO.hpp:155-161`: `isAutoregName` only matches KV/past/cache/position_id names.
- `TextToken.cpp:117`: `refused = isAutoregressive(io)`.

So decoder_iter (classified TextToken today) is **loaded**. `resolveIO` then picks the bool `mask` as the token input (it is the only TokenSeq), so the model fails at run time and hits X3's sticky flag. It is never refused.
Wrong text in score-master `docs/MODELS.md`:
- **L570** ("No compatible object" table, Why column): "autoregressive Tacotron2 decoder step; Text Token Processor refuses it (tests/classify_real_models.py)".
- **L460**: "**Refuses** autoregressive/KV-cache graphs." This is only true for KV-cache names.
- The same false claim is in `tests/classify_real_models.py` EXPECTED: `"tacotron2__decoder_iter.onnx": ("TextToken", True),  # autoregressive: node refuses at load`.
- Unrelated stale item: **L481-485** EmotionNet says "*No compatible model found on either drive*". `/mnt/win2/models/models-presets/models/emotionnet/enet_b0_8_best_{afew,vgaf}.onnx` exist, and `image-processor/hsemotion-enet-b0-8.onnx` matches too. The preset folder is probably newer than the scan.

Fix (docs):
- L570 Why → "Autoregressive Tacotron2 decoder step (recurrent attention/decoder LSTM state threaded by the caller). **Not refused today**: `isAutoregressive` only matches KV-cache names (ledger T7), so Text Token Processor loads it, reads the bool `mask` as tokens, and fails at run time. No object drives the per-step loop."
- L460 → "Refuses KV-cache graphs (`past_*`/`present*`/`cache` names). Recurrent-state decoders (tacotron2 `decoder_iter`, pocket-tts `lm_main`, moonshine `cached_decode`) are not detected yet (T7)."
- Rewrite both lines again once T7 lands.
- Add the EmotionNet rows.

Risk: none (docs). Test: n/a. Effort: S

---

### E1 — EmotionNet keeps the stale session on a model change
Status: CONFIRMED (see the X2 repro: afew → ferplus gives identical output). Root cause, fix and test: as X2 at `ENet.cpp:37`. Effort: S

### E2 — The resolution knob and the model's input shape disagree
Status: PARTIAL. The ledger says any value other than 224 throws. In fact only a *smaller* knob throws; a *larger* knob silently feeds garbage, and dynamic H/W models always throw.
Evidence: `Images.hpp:81-83` copies `port.shape` (only `[0]` is forced to 1), `:88` sizes the buffer as `3*model_w*model_h` from the knob, and `Utilities.hpp:72-77` calls `CreateTensor(data, data.size(), shape)`. `$S/e2`:
```
buffer 224² for shape [1,3,224,224]: created OK
buffer 256²:                         created OK   <- ORT only checks buffer >= shape; reads the first 3·224² floats of
                                                     256²-strided planes → G/B planes misaligned, garbage input, no error
buffer 128²:                         THROW not enough space: expected 602112, got 196608  → sticky invalid (X3)
shape [1,3,-1,-1]:                   THROW tried creating tensor with negative value in shape
```
Root cause: the tensor shape comes from the model and the buffer from the knob, and nothing reconciles them.
Fix:
- In ENet/Resnet, derive the size from the model: `s = spec.inputs[0].shape; mw = (s.size()==4 && s[3]>0) ? s[3] : knob.x; mh = (s[2]>0) ? s[2] : knob.y;`.
- In `nchw_tensorFromRGBA`, set `input_shape[2]=model_h; input_shape[3]=model_w` for rank 4. This fixes dynamic-H/W models, and a knob mismatch then errors in ORT instead of garbling.
- Keep the knob, but only for dynamic-H/W models.

Risk: `nchw_tensorFromRGBA` is also used by YOLO-pose/blob/segmentation, TRTPose and `FastVLM.cpp:54`. Writing the knob into the shape only changes behaviour when they already mismatched.
Test: a `x2.cpp` variant: res 256 must equal res 224 output; a dynamic-H/W export must run.
Effort: S

### E3 — 1-channel models (FER+ `[1,1,64,64]`)
Status: PARTIAL. They do not throw at any knob ≥ 64, but get the wrong input **and the wrong labels**.
Evidence: `$S/x2` output: `fresh ferplus res=64: invalid=0 [Anger 0.748] ...`, `res=224: invalid=0 [Anger 0.749] ...`.
- The buffer is 3 planes, so ORT takes the first 64² floats: the **R plane only**, ImageNet mean/std-normalised. At 224 it takes the top ≈18 rows of the R plane.
- FER+ expects grayscale 0-255, unnormalised (ONNX model-zoo contract).
- Its 8 outputs are ordered neutral, happiness, surprise, sadness, anger, disgust, fear, contempt. The node labels index 0 as "Anger" (`classes_8`, `Resnet.hpp:94-103`, AffectNet order). So a FER+ "neutral" shows as "Anger".

Root cause: a single RGB/ImageNet preprocessing path and a single label table.
Fix:
- If `shape[1]==1`, fill a luma plane in 0..255 with mean 0 and std 1. `ImageOps` already has `TensorLayout::NchwGray` (`ImageOps.hpp:70`, `ImageOps.cpp:101/191`).
- Pick the label table by model family, e.g. by input `[1,1,64,64]`, or add a labels-file port like Resnet has.
- Alternatively, drop FER+ support and document EmotiEffLib-only.

Risk: ENet only. Test: FER+ on a face crop gives the right top class. Effort: M

### E4 — Face-crop classifiers run on the whole frame
Status: CONFIRMED (design): `ENet.cpp:44-53` resizes the whole texture (`resize_fill_crop_rgba`). There is no ROI input and no detector.
Fix: cheapest is to document that the face must fill the frame, and suggest Pose Detector (face) → crop upstream. Proper fix: an optional ROI/rect input, or reuse PoseDetector's BlazeFace stage. Effort: S (doc) / L (detector)

### E5 — 10-class output: padding labels and the `min(N,8)` cap
Status: CONFIRMED by reading the code; not run, since no 10-output ("mtl") EmotiEffLib model exists on the drives (searched `*mtl*`, `enet_b*`).
Evidence: `Resnet.hpp:104-106` pushes `"class_8"`, `"class_9"`; `:133` softmaxes all N; `:136` has `for (i < std::min(N, 8))`.
Root cause (the ledger undersells it): in EmotiEffLib the 10-output `*_va_mtl` models are **8 emotion logits + valence + arousal**. I did not re-check the upstream source here; this is from memory of `facial_analysis` (`scores[:, :-2]`).
So the current code (a) softmaxes valence/arousal into the emotion distribution, which corrupts all 8 probabilities, and (b) drops V/A. N=9 (7 emotions + V/A) is rejected at `:123`.
Fix: if N is 10 (or 9), softmax only the first N-2 values and emit the last two raw, as "Valence"/"Arousal" entries or separate outputs. Size the loop from the label count.
Risk: ENet only. Test: a synthetic 10-logit tensor through `processOutput` (the `r3.cpp` harness style). Effort: S

### E6 — Wrong metadata
Status: CONFIRMED: `ENet.hpp:30-35` has `author "Resnet authors, Onnxruntime"`, `description "Resnet recognizer using a DNN model."` and `manual_url …#resnet`.
Fix: description "Facial emotion recognizer (EmotiEffLib / HSEmotion EfficientNet)", author "EmotiEffLib authors, Onnxruntime", and a `#emotionnet` anchor (check the docs site). Effort: S

### R1 — Resnet keeps the stale session on a model change
Status: CONFIRMED (the X2 repro: resnet18 → alexnet gives identical top-5). Fix as X2 at `Resnet.cpp:26`. Effort: S

### R2 — Resnet's invalid flag is sticky
Status: CONFIRMED (the X3 repro, run on ResnetDetector: res 128 → 224 → same file re-selected all stay `invalid=1`). Fix as X3 at `Resnet.cpp:63-66`. Effort: S

### R3 — `thread_local idx(N)` is sized once
Status: CONFIRMED, and worse than described. The ledger says it is shared with ENet (L128); that is **wrong**: `EmotionNet::processOutput` has no `idx`. Two further problems:
- The vector is `thread_local`, so two Resnet nodes with different N on the same render thread corrupt each other **without any model change**.
- There is an out-of-bounds read for N < 5.

Evidence: `Resnet.hpp:53-68`. `$S/r3` (built with the default Arch `_GLIBCXX_ASSERTIONS`):
```
shrink (N=1000 then N=8): stl_vector.h:1253 vector<float>::operator[] Assertion '__n < this->size()' failed  (recog[i1], i1 up to 999)
small  (N=2):             vector<int>::operator[] Assertion failed  (idx[i] for i<5)
grow   (N=8 then N=1000): N=1000: c0=2e-09 c1=2e-09 ...  <- true top class c900 (p≈1) never ranked: idx has only 8 entries
```
Without assertions: the shrink case reads stale `recog` capacity and can output class indices ≥ N; the small case reads past `idx`.
Root cause: the `thread_local` initialiser runs once per thread, and the loop bound is a hard-coded 5.
Fix:
```cpp
thread_local std::vector<int> idx; idx.resize(N); std::iota(...);
const int k = std::min(5, N);
std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), cmp); for (i < k) ...
```
Risk: Resnet only. Test: `r3.cpp` shrink/grow/small asserting top-1 = argmax. Effort: S

### R4 — Softmax applied to outputs that are already probabilities
Status: CONFIRMED by reading the code (`Resnet.hpp:51`, `EmotionNet` at `:133`). No shipped preset is affected: all five resnet presets and both ENet presets output logits (random input: sums −3.4…14.8, negative mins).
For a model that does end in Softmax, the values are crushed to ≈1/N: e.g. p_top=0.9 over 1000 classes → e^0.9/(e^0.9+999) ≈ 0.0025. The ranking is preserved.
Fix: skip the softmax when `min ≥ 0 && |Σ−1| < 1e-3`.
Risk: nil (a logit vector essentially never satisfies both). Test: synthetic probability vector through `processOutput`. Effort: S

### R5 — Resolution knob vs model shape (Resnet)
Status: PARTIAL, as E2. Reproduced on ResnetDetector: alexnet at res 128 throws and stays invalid. A larger knob silently garbles. `[N,3,-1,-1]` exports always throw. Same fix as E2 (`Resnet.hpp:50` knob, `Resnet.cpp:33-42`). Effort: S


## Diagnosis: image-video group (I1–I4, I5 HAN, V1–V4)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx` (nothing modified).
Scratch: `docs/diagnosis-repros/image-video/`. The repro scripts are `layout.cpp`, `arch.cpp` (clang++ against the real headers), `seg.py`, `han2.py`, `rvm.py` and `io.py`. Images: `han_cmp.png`, `seg_cmp.png`, `dexined_outs.png`, `dexined_sig.png`, `rvm_*.png`.
Score was not launched.

---

### I1 — 4-channel image inputs throw and the model is marked invalid for good
Status: CONFIRMED
Evidence:
- `ImageModelRole.hpp:74-76`: `isChan(d)` accepts 4, so `[1,4,512,512]` resolves to `NchwRgb` with ch=4 (layout.cpp prints `layout=1 ch=4`).
- `ImageProcessor.cpp:577-578,614`: `channels = (layout==NchwGray)?1:3` and `ishape = {1,3,mh,mw}`. The same happens in `preprocessTexture` (L425, L456) and `VideoProcessor.cpp:627,662`.
- Repro (`seg.py`), 060__hair_segmenter.onnx fed the 3-channel tensor that the C++ builds: `INVALID_ARGUMENT : Got invalid dimensions for input: input`. That throws out of `operator()`, and `catch(...)` sets `current_model_invalid = true` (L546-549). The flag is sticky.
- The same model fed 4 channels (RGB/255 plus alpha=0) runs fine: about 10 % hair, and a clean hair mask in `seg_cmp.png`. On frame 2, feeding the previous mask as the 4th channel changes the result by only 0.011 mean abs, so a zero prior is acceptable.

Root cause: the ImageOps samplers only ever emit 1 or 3 planes. The declared channel count (`role.in_channels`) is never used to size the tensor.

A second blocker on the same model: its output is NHWC `[1,512,512,2]`. `resolveLayout` does not treat 2 as a channel count (`isChan` = {1,3,4}), so it falls through to "symbolic channel → NCHW" and reads ch=512, h=512, w=2 (layout.cpp). Fixing the input alone would therefore produce a 2×512 garbage mask. This needs the NHWC-2ch part of I2.

Fix:
- `runImage`, `preprocessTexture` and VideoProcessor `runImage`: when `role.in_channels == 4`, sample 3 channels as today into a temp buffer (or into the first 3 planes), then add the 4th channel:
  - NCHW: `storage.resize(4*HW)`, and fill plane 3 with 0, or with the Aux texture's R channel normalized the same way when Aux is connected.
  - NHWC: re-interleave RGB→RGBA.
  - `ishape` gets C=4.
- Sketch (NCHW): `if(inC==4){ storage.resize(4*HW); std::fill(storage.begin()+3*HW, storage.end(), 0.f); ishape[1]=4; }`. NHWC needs a small repack loop.
- Also: do not set `current_model_invalid` on a shape-mismatch exception (general X3 issue).

Risk / blast radius:
- NCHW-4 inputs are currently broken anyway, so there is no regression.
- 4-channel RGBA models that want real alpha get 0 unless the fix uses source alpha. The texture is RGBA, so `alpha/255` is an option. Choose zero or Aux for the prior-mask models, and document it.

Test: extend `tests/image_node_test` (or the py harness) with 060__hair_segmenter. Expect no exception, and a mask decoded at 512×512 once the I2 NHWC fix is in.
Effort: S (input). M together with the NHWC-2ch output fix.

---

### I2 — 2-class outputs: Mask takes channel 0 (background), so the mask is inverted
Status: CONFIRMED, and worse than the ledger says (three distinct failures)
Evidence:
- `TensorToTexture.cpp:128-155` `writeMaskImpl` always reads channel 0 (`data[p]` / `data[p*C]`).
- **PP-HumanSeg** (`pinto/196__human_segmentation_pphumanseg_2021oct.onnx`, out `[1,2,?,?]`):
  - `classifyImage` gives ImageToMask (layout.cpp).
  - The output is already softmaxed: person region ch0 = 0.121, ch1 = 0.879; background ch0 = 0.999, ch1 = 0.001.
  - So ch0 is the inverted mask (`seg_cmp.png`, tiles 2 and 3).
- **hands_segmentation_pytorch** (`wild2/…hands_segmentation_pytorch.onnx`, out fully symbolic):
  - The symbolic channel defaults to 3 (`ImageModelRole.hpp:103`), so the load-time kind is **ImageToImage**.
  - At runtime the output is `[1,2,256,256]` logits: ch0 mean +15.8 (bg), ch1 mean −15.9.
  - `effectiveKind` (ImageProcessor.cpp:213-218) only demotes channels == 1, so on Auto it goes to `writeRgb` with C=2 (R=ch0, G=ch1, B=ch0), a magenta/green picture, not a mask.
  - With Task=Mask it is inverted.
- **hair_segmenter 060**: NHWC `[1,H,W,2]` is misparsed as NCHW ch=H, w=2 (see I1).

Root cause:
- The mask decode has no notion of "foreground class".
- 2 is not a recognised channel count for NHWC detection.
- C=2 is not treated as a mask.

Fix (all in helpers, shared by both nodes):
1. `ImageModelRole.hpp` `resolveLayout`: for rank 4, when neither `s[1]` nor `s[3]` is in {1,3,4}, check `s[3]==2` → NHWC, then `s[1]==2` → NCHW, before the symbolic fallback. Put this in the rank-4 branch only, so that `[N,H,W]` rank-3 masks are unaffected.
2. `effectiveKind`: `if(os.spatial && os.channels==2 && kind==ImageToImage) return ImageToMask;`
3. `writeMaskImpl`: when `C==2`, compute the foreground value per pixel: `v = (min≥0 && max≤1 over ch1) ? ch1 : 1/(1+exp(ch0-ch1))`. This is the softmax of the 2 logits. Decide per frame from a min/max pass over ch1. Then apply the WriteMode (DirectClamp is correct for both cases). Same change in `writeMaskF`.
   - Optional: a "Mask Channel" spinbox for C>2 class maps. Not needed for the 2-class case.

Risk / blast radius:
- `resolveLayout` is used by `classifyImage`, `makeOutSpec` and `preprocessTexture` in both nodes.
- The only shapes whose result changes are those with a literal 2 in slot 1 or 3 and no 1/3/4 in the other. Today those are all misparsed anyway: FILM flow `[.,.,.,2]` and optical-flow outputs will now parse as 2-ch NHWC and decode as a "mask" of the flow's softmax. That is meaningless but no longer garbage-shaped. Flow models should use Data or a future flow visualisation.

Test:
- py harness plus a C++ unit on `makeOutSpec({1,512,512,2})` → nhwc, ch=2.
- PP-HumanSeg / hand-seg: mean(mask) over the person region > 0.5.

Effort: M

---

### I3 — rank-2 `[?,1]` scalar typed Latent; input 0 not an image → LatentToImage path
Status: CONFIRMED (code path and classifier run; FILM was not executed end-to-end)
Evidence:
- `arch.cpp` against the real `ModelArchetype.hpp` on FILM's I/O: `time [?,1]` → arch 7 (Latent); x0/x1 → Image. The Scalar branch is `rank <= 1` only (`ModelArchetype.hpp:150-151`). Rank 2 goes to `if(is_input) return PortArchetype::Latent;` (L200-201).
- `classifyImage(io)` uses `io.inputs[0]` (`ImageModelRole.hpp:121`). For FILM, input 0 is `time`, so the kind is `LatentToImage` with latent_dim=1 (layout.cpp).
- `ImageProcessor::operator()` L540-542: the LatentToImage check comes **before** `runImage()`, so `multi_input` is never reached.
- `runLatent` builds a single `[1,1]` tensor and `dispatchInfer(... force_async=true)` passes `ins[1]` to a 3-input model. ORT throws "missing input" in the worker, the `catch(...)` there only clears `inferenceInProgress`, and `produced_latent` short-circuits every later frame. The result: silently no output, and the model is not even marked invalid.
- Even if routing were fixed, `runImageMulti` would zero-fill `time` (not a Param), so the output is t=0, i.e. roughly x0.

Root cause: input roles come from index 0 rather than the archetype, and `[1,1]` or `[?,1]` is not seen as a scalar.

Fix:
1. `ImageProcessor::reloadModel`: after computing `image_input_index` from `march`, build the ImageIO with that input first, as VideoProcessor's `toImageIO(spec, image_in)` does (`VideoProcessor.cpp:106-114`). Then compute `role` and `out_kinds` from it.
2. Only take `runLatent` when `march` has **no** Image input.
3. In the Param binding loop (L488-503), also accept `a == Latent` when the flattened concrete size is 1, i.e. every dim is ≤1 or symbolic and rank ≤ 2. This is local to ImageProcessor and avoids touching ModelArchetype (whose Latent semantics other nodes rely on).

Risk / blast radius:
- ImageProcessor only.
- GANs with a real latent keep input 0 = latent and no Image input, so they are unchanged.
- The Param binding change only affects models with a `[1,1]` input.

Test: FILM (`wild/film__film_net.onnx`): expect `multi_input=true`, `param_in[0]=0`, `image_input_index=1`, `aux=2`, and output 26 (`image`) decoded to Image. The output index must be set, or rely on I4-style routing.
Effort: S–M

---

### I4 — multi-output models need an Output Index that Auto doesn't find (DexiNed)
Status: CONFIRMED (mitigated by preset), plus a new finding on the pixel mapping
Evidence:
- `dexined.onnx` outputs: `out_1..out_6` (index 0–5) and `block_cat` (6), all `[1,1,h,w]`.
- With index 0, `decodeAll` sends out_1 to Mask. Outputs 1–6 are also Mask-kind, so they are skipped (the target is taken). The fused map is never shown.
- `dexined_sig.png`: sigmoid(out_1) is noisy and textured; sigmoid(block_cat) gives clean edges.
- The preset already sets Output Index 6 (`presets/Image Processor/DexiNed edge detection.scp`: `[7, {"Int": 6}]`).
- New finding: the preset uses `MinMaxNormalize` on logits in the range −27..4. The background logit (about −5) maps to about 0.7 grey, so the result is washed out (`dexined_outs.png`, last tile). DexiNed expects sigmoid.

Root cause: there is no semantic ordering among same-kind outputs. The fused output is last for DexiNed but first for U²-Net (`u2net.onnx` outputs are all numeric names `1959..1965`, with d0 = index 0). So neither "first" nor "last" is right, and a name heuristic only works for DexiNed (`block_cat`).

Fix:
- Keep the Output Index as is, and document it. Recommend no heuristic: "prefer last" breaks U²-Net, and "fuse"/"final" names don't match DexiNed's `block_cat`.
- If one is wanted anyway: in `reloadModel`, when `inputs.output_index` is 0 and more than 4 outputs share one kind, pick the first output whose name contains `fuse`, `final` or `block_cat`. This is hacky; prefer the preset.
- Separately, add a `Sigmoid` WriteMode (`255/(1+exp(-x))`) to `TensorToTexture` and `OutputMode`, and switch the DexiNed preset to it. The same primitive helps I2's logits case.

Risk / blast radius: a new enum value in `OutputMode` must be appended at the end, to keep saved scores stable.
Test: DexiNed on a photo. With Sigmoid, the background is below 10/255 and the edges are above 200/255.
Effort: S (Sigmoid mode plus preset). Documentation only for the index.

---

### I5 (HAN x3 part) — preset None/Passthrough vs Centered/Denormalize
Status: DECIDED — keep the preset (None/Passthrough). The 31.2 dB sweep result is an artefact.
Evidence (`han2.py`, `han_cmp.png`):
- The earlier 31.2 dB came from the `sr` test image, which is only 256×256. Its "HR" was an upscale, so PSNR rewards blur. With native HR crops (480², ×3 bicubic down):

| image | bicubic | None/Passthrough | Centered/Denorm | DivBy255/DirectClamp |
|---|---|---|---|---|
| modnet | 27.70 | 23.25 | 25.66 | 23.42 |
| aeroplane | 28.90 | 25.37 | 26.66 | 24.06 |
| mtcnn | 24.81 | 22.23 | 23.72 | 22.21 |
| animalpose | 30.18 | 26.19 | 26.83 | 24.17 |
| demo1 | 27.67 | 22.58 | 25.67 | 23.57 |

- PSNR is useless here (bicubic wins everything). By eye (`han_cmp.png`, cols: HR | bicubic | None/Passthrough | Centered/Denorm):
  - **None/Passthrough is a real ×3 SR**: sharp hair strands, crisp aircraft edges, cow fur, with some over-sharpening halos. The raw output range is −57..340, so it clips.
  - **Centered/Denormalize is visibly pixelated/blocky and low-contrast**, i.e. the network is out of its input domain.
- HAN is an EDSR-family model with `rgb_range=255` and an internal mean-shift, so 0..255 input is canonical. BGR vs RGB makes a difference of 0.1 dB or less.

Root cause: none. The model is working as designed; the PSNR metric misled the earlier sweep.
Fix: none; close the HAN part of I5. Optionally note in the ledger that SR PSNR sweeps must use a native-resolution HR, not the 256² `sr` image.
Effort: —

---

### V1 — VideoProcessor decodes only `outs[output_index]`
Status: CONFIRMED
Evidence:
- `VideoProcessor.cpp:418-419`: `const int idx = std::clamp(output_index, 0, nout - 1); return decodeOutput(outs[idx], kind, wm, scratch);`
- `runInferAndThread` returns a single `DecodedOutput`, and `VideoInferJob` carries only `output_index`.
- For RVM (`fgr, pha, r1o..r4o`), one node gives fgr or pha, never both.

Root cause: the I7 `decodeAll` fix was applied to ImageProcessor only.

**What can be shared.** All of it lives in ImageProcessor.cpp's anonymous namespace today, and VideoProcessor.cpp holds a stale near-copy:
- Identical, and can move verbatim into a shared header/TU (say `OnnxModels/ImageDecode.{hpp,cpp}`, namespace `OnnxModels::imgdec`):
  - `DecodedOutput` (IP L193-208 == VP L224-232)
  - `effectiveKind` (IP L213-218; VP inlines the same test at L247-248)
  - `targetOf` (IP L221-238)
  - `decodeOutput` (IP L240-287 == VP L234-282)
  - `DecodedOutputs` and `decodeAll` (IP L329-373)
  - `resolveWriteMode` and `applyTaskOverride` (identical in both)
- `applyDecoded`: make it a template on the node type. Both nodes have the same `outputs.{image,mask,depth,data}` members. Take IP's version (L289-327), which has the 0-size guard and the `r8.empty()` check that VP's L284-314 lacks. VP's Depth branch writes Mask unconditionally.
- Also duplicated, and could share a `ImagePreprocess.hpp`: `floatToHalf`, `normConstants`, `toTensorLayout`, `snapDim`, `buildInputTensor`, `isHeavyModel`, `concreteShape`. Not required for V1.

**VideoProcessor-specific change:**
- Compute `out_kinds` in `reloadModel` from `toImageIO(spec, imageInputIndex)`.
- Then set `out_kinds[s.output_index] = Unknown` for every state slot, so `targetOf` returns None. This is **mandatory**: `classifyImage` labels RVM `r1o..r4o` (16/20/40/64 ch) as ImageToImage (layout.cpp prints `rvm out 2..5 kind=ImageToImage`). With Task=Mask on index 0, the Image outlet is left free and `decodeAll` would decode the 16-channel state r1o as RGB into Image.
- Add `std::vector<ImageModelKind> out_kinds` to `VideoInferJob`. A vector assignment reuses the recycled capacity, as InferJob does.
- `runInferAndThread` returns `DecodedOutputs` via `decodeAll(outs, idx, kind, wm, out_kinds, scratch)`. The worker lambda captures the `DecodedOutputs`.

Result for RVM:
- index 0 (fgr): Image = fgr; pha goes to Mask with the Auto mapping (MinMax today, DirectClamp after V2).
- index 1 (pha): Mask = pha; fgr goes to Image with DirectClamp. That is correct for RVM's [0,1] fgr.

Risk / blast radius:
- VideoProcessor only, plus a pure refactor move in ImageProcessor.
- Extra cost is one secondary `writeRgb`/`writeMask` pass per frame on the worker thread (about 2 Mpx at HD). It is only paid when the outlet is free.
- `applyDecoded` now touches two outlets per frame. The texture outputs are independent.

Test:
- `tests/video_node_test.cpp` (exists, untracked): assert that after one RVM frame both `outputs.image` and `outputs.mask` changed, and that neither was written from an r#o tensor (the Image size equals the src size and 3-ch content is in [0,255]).
- `tests/validate_rvm_loop.py` for the numerics.

Effort: M

---

### V2 — Auto on Mask gives MinMaxNormalize and stretches the alpha matte
Status: PARTIAL (the mechanism is confirmed; the effect is smaller than described)
Evidence:
- `VideoProcessor.cpp:138-142` (and `ImageProcessor.cpp:134-139`): Auto gives DirectClamp only for ImageToImage, otherwise MinMaxNormalize.
- RVM, 512×288, dr=0.25 (`rvm.py`):
  - Truly empty frames (wall crop, uniform grey, depth scenes) give pha **exactly 0**. MinMax sees a flat frame, `inv_range=1`, so the output is all black. There is no noise explosion there.
  - Frames with weak false positives: pha max 0.05 (rotnet), 0.21 (cow), 0.32 (aeroplane), 0.43 (3D-det demo). MinMax promotes those few pixels (≤0.1 %) to 255, and the stretch factor changes every frame, so in video it flickers as white specks.
  - With a person present, the range is already 0..1, so MinMax is a no-op.
- All 7 RVM presets set `DirectClamp` explicitly, so only Auto users are affected.
- The issue becomes more visible after V1: secondary outputs always use Auto (`decodeAll` L366), so a secondary pha would be min-max stretched.

Root cause: the Auto mapping for masks assumes an unknown range, even when the tensor is already a probability.

Fix: add a runtime-range Auto for single-channel outputs, in `TensorToTexture.cpp`.
- A new `WriteMode::AutoRange`, or handle it inside `MinMaxNormalize`'s pre-pass.
- `writeMaskImpl` already computes lo/hi. If `lo >= 0 && hi <= 1`, use DirectClamp mapping (`v*255`); otherwise min-max.
- `resolveWriteMode(Auto, non-image)` returns `AutoRange`. `decodeOutput` keeps `MinMaxNormalize` for the Depth visual.
- Make the change in the shared file (see V1), so ImageProcessor gets it too: u2net and isnet sigmoids then become DirectClamp, and DexiNed/depth logits stay min-max.

Risk / blast radius:
- Both nodes' Auto mask decode.
- A mask model whose sigmoid output never spans the full range (e.g. 0.3..0.6) would look dimmer than today. That is arguably more correct.

Test: unit-test `writeMask` with a synthetic [0,0.3] plane: AutoRange gives max 76, not 255. A [−5,5] plane is still stretched.
Effort: S

---

### V3 — Reset ignored while a job is in flight
Status: CONFIRMED (code path; not exercised in score)
Evidence:
- `VideoProcessor.cpp:584-585`: `if(inputs.reset && !inferenceInProgress) zeroStates();`. The impulse is not latched.
- `dispatchInfer` L680-682 sets `inferenceInProgress = true` in the same tick that follows a completion.
- The completion lambda (L748-756) overwrites `self.states` with the job's chain (`self.states = std::move(states)`).
- Measured job times (CPU, default threads): HD dr=0.25 takes 192 ms and 4K dr=0.125 takes 414 ms, i.e. about 12–25 render frames at 60 fps per job.
- `inferenceInProgress` is false only during the operator() call right after a completion. So a Reset lands only if it arrives on that exact tick: roughly a 1/12 to 1/25 chance. Otherwise it is lost, and the old recurrent chain continues.

Root cause: a one-tick impulse is gated by a busy flag with no latch.

Fix: add `bool resetPending = false;` to VideoProcessor.
- In `operator()`: `if(inputs.reset) resetPending = true; if(resetPending && !inferenceInProgress) { zeroStates(); resetPending = false; }`
- When a job completes, its lambda first sets `inferenceInProgress=false` and adopts the stale states. The next `operator()` call then zeroes them before `runImage()` dispatches, which is correct.
- Optionally drop the stale adoption in the lambda when `resetPending` is set. That is the same thing, one step earlier.

Risk / blast radius:
- Only the Reset path.
- The geometry re-zero at L612-622 already retries every frame (it doesn't update `stateGeom*` while busy), so it is not affected.
- Unrelated weakness noticed: the lambda's model-swap guard is only `self.states.size() == states.size()`. A swap to another model with the same number of state slots would adopt foreign states, and ORT would then throw. `reloadModel` should bump a generation counter that the job carries and the lambda checks.

Test: in `video_node_test.cpp`, drive the node with a stub worker that defers completion. Fire Reset while `inferenceInProgress`, complete the job, tick once more, and assert that every `states[i].buffer` is all-zero with its declared initial shape.
Effort: S

---

### V4 — 512×288 counts as "not heavy" and runs synchronously on the render thread
Status: CONFIRMED (and the cause is `downsample_ratio`, not size)
Evidence:
- `VideoProcessor.cpp:316-319`: heavy means `model_bytes > 64 MiB || in_pixels > 512*512`. RVM is about 15 MB, and 512×288 = 147 k px, so the job runs sync.
- `rvm.py` timings, on this 32-core box:

| setting | default threads | 4 threads | 1 thread |
|---|---|---|---|
| 512×288, dr=1.0 | 160 ms | 147 ms | 194 ms |
| 512×288, dr=0.25 | 14.5 ms | 16.4 ms | 16.6 ms |

- The preset "RVM Matte 512 fast" sets Param 1 (downsample_ratio) to **1.0**, which is RVM's recommended value at ≤512. So the preset that is named "fast" blocks the render thread for about 150–190 ms per frame.

Root cause: the heavy heuristic is static (file size and pixel count) and ignores inputs that dominate cost (downsample_ratio), as well as the EP/CPU speed.

Fix: measure instead of guessing, in both nodes' `dispatchInfer`.
- Time the sync `infer+decode` with `std::chrono::steady_clock`.
- If it exceeds a budget (e.g. 8 ms), set a sticky `preferAsync = true`.
- Reset `preferAsync` in `reloadModel` and when the geometry or params change (VideoProcessor already tracks `stateGeom*`).
- `heavy = isHeavyModel(...) || preferAsync;` So at most one slow frame happens per configuration.
- Sketch:
  ```cpp
  auto t0 = std::chrono::steady_clock::now();
  /* ...sync run... */
  if(std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(8))
    preferAsync = true;
  ```

Risk / blast radius:
- Switching VideoProcessor to async adds one frame of latency and latest-wins drops. The state chain stays coherent, because the job carries and returns the states.
- ImageProcessor, if changed too: same latency trade-off.

Test: RVM at 512×288 with dr=1.0. After the first frame `preferAsync` becomes true, and later `operator()` calls return in under 2 ms. This can be asserted in `video_node_test.cpp` with a steady_clock around `operator()`.
Effort: S

---

## Extra findings (not in the ledger)
- **I4b / DexiNed preset:** MinMaxNormalize on logits gives a grey, washed-out result. Needs a `Sigmoid` WriteMode (see I4).
- **NHWC 2-channel outputs are misparsed** (`resolveLayout` fallback reads ch=H, w=2). This affects hair_segmenter 060, FILM flow outputs and any NHWC 2-class seg (see I2 fix 1).
- **VideoProcessor `applyDecoded`** lacks ImageProcessor's 0-size guard (VP L284-314 vs IP L293-294). A degenerate output creates a 0×0 texture and memcpys from an empty buffer. This goes away if V1 shares IP's version.
- **Job model-swap guard** in VideoProcessor compares state counts only (see V3).


## Diagnosis — group `audio` (A1–A6, AA1–AA4)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx` (identical to score-master for AudioProcessor.cpp, AudioAnalyzer.cpp,
AudioIO.hpp, ModelArchetype.hpp, SequenceProcessor.cpp — checked with `cmp`). Score was not launched. Nothing in the tree was modified.

Scratch repros (all in `docs/diagnosis-repros/audio/`):
- `io.py` — dumps real model I/O signatures (onnx).
- `pairing.cpp` — compiles against the real `ModelArchetype.hpp` / `AudioIO.hpp`, runs `classifyModel` on the real signatures and
  replicates verbatim the state-pairing loops of AudioAnalyzer (L110-136), AudioProcessor (L149-177), SequenceProcessor (L200-230),
  plus the wave in/out selection and `WaveformShape::fromInputShape(...).tensorShape(...)`.
- `a1sim.cpp`, `a1sim2.cpp` — AudioProcessor async path (operator/runBlock/dispatchInfer) over the real `WaveformInput/Output`, fake
  identity model with latency, current vs proposed fixes.
- `silero.py`, `demucs.py` + inline ORT checks (CREPE, CLAP, hifigan, dtln).

Real signatures used (from `io.py`):
```
silero-vad-v4-16k / wild silero_vad : IN input[b,seq] sr[] INT64 h[2,b,64] c[2,b,64]  OUT output[b,1] hn[2,b,64] cn[2,b,64]
silero-vad-512 (k2-fsa/sherpa)      : IN x[1,512] h[2,1,64] c[2,1,64]                OUT prob[?,1] new_h[2,?,64] new_c[2,?,64]
crepe-full                          : IN input[n,1024]                                OUT output[?,360]
CLAP fusion (wild2)                 : IN longer[1,1] BOOL  mel_fusion[1,4,1001,64]    OUT text_embed[1,512]
demucs htdemucs_ft_vocals (289 MB)  : IN mix[1,2,l]                                   OUT x[?,?,?,?]  (runtime [1,4,2,N])
dtln1 : IN input_2[1,1,257] input_3[1,2,128,2]  OUT activation_2[1,1,257] tf_op_layer_stack_2[1,2,128,2]
dtln2 : IN input_4[1,1,512] input_5[1,2,128,2]  OUT conv1d_3[N,1,512] tf_op_layer_stack_5[N,2,128,2]
hifigan generator_dynamic : IN input[1,80,T]   OUT output[1,1,T']
deep-music-enhancer-resnetbn-stereo-8192 (210 MB) : IN x[b,2,8192] OUT y[b,2,8192]
```
No model carries a sample-rate in `metadata_props`.

---

## Audio Processor

### A1 — async path consumes and discards input blocks while a job is in flight
Status: CONFIRMED (with a nuance: it bites whenever job latency incl. the main-thread hop exceeds one block period, including transient spikes, not only for "slow models")
Evidence:
- `AudioProcessor.cpp:234-235` `while(audio_in.ready() && guard++ < 32) runBlock();`
- `:249` `const int64_t n = audio_in.fill(staged);` (fill = peek + `drop(hop)`, AudioIO.hpp:379-390) happens **before**
- `:272-275` `if(force_async) { if(inferenceInProgress) return; ...`
- Result delivery path (`score-plugin-avnd/Crousti/Executor.hpp:891-936`): worker thread → `ossia::qt::run_async(qApp, …)` (main thread) →
  `worker_results.enqueue` → drained at the start of the node's next `run()` (Executor.hpp:85-99). So job latency = inference +
  main-thread event-loop latency + up to one tick.
- Input ring capacity is only one block + one host buffer (`AudioIO.hpp:320-322` `cap = block + per_block + 2`), and output ring
  one block + one host buffer (`:418-420`).
- `a1sim2` (block/latency in 512-frame ticks @48k):
```
block=8192 (171 ms), job 107 ms, +150 ms spike every 20 jobs
  CURRENT     dispatched  95.0% of input, blocks consumed-and-dropped=8,   output non-silent 94.6%
  GATE        dispatched  99.6%,           dropped=0,                       output non-silent 98.6%
block=1024 (21 ms), job = 2-tick hop + 1-tick inference
  CURRENT     dispatched  50.0%, dropped=703,  output 49.9%
  GATE        dispatched  66.6%, dropped=0,    output 66.6%
  GATE+BATCH  dispatched 100.0%, dropped=0,    output 99.8%
```
- Demucs measured in ORT: `1024 -> (1,4,2,1024) 10.91s`, `343980 -> 11.37s` (it pads to its segment internally), so with the
  default 1024-sample block (dynamic `l` → 1024) Demucs keeps ~0.2% of the input.
Root cause: the block is popped from the ring before the in-flight check, so every block that becomes ready while a job is pending
is thrown away (the `while` loop even drains up to 32 of them in one tick). There is no backlog capacity in the ring and one block
per job means the per-job fixed cost (worker post + Qt main-thread hop + tick quantisation) is paid per block, so small blocks
cannot keep up even with a fast model.
Fix (all on the audio thread unless noted; `inferenceInProgress` is only read/written on the audio thread — set in dispatch, cleared
by the result lambda that runs in `node_with_worker::run` on the same thread — so a plain bool stays correct, no atomics needed):
1. Gate before consuming. In `operator()`: compute `heavy` once per tick and
   `while(audio_in.ready() && guard++ < 32) { if(heavy && inferenceInProgress) break; runBlock(); }` — i.e. never call `fill()` while
   a job is pending. Lock-free, no allocation, samples stay in the ring.
2. Give the ring backlog: in `resolveIO` pass a larger capacity to `WaveformInput::prepare` for async models (e.g.
   `block * (1 + K) + per_block`, K≈4; add a `backlog_blocks` parameter rather than abusing `max_host_frames`). Overrun keeps the
   existing `AudioRing::push` latest-wins semantics, which is then the honest "model slower than real time" case.
3. Batch all ready blocks into one job (needed for small blocks): when dispatching, loop `fill()` up to K times appending into
   `job->input` (k·C·N) with `job->nblocks = k`; `work()` runs the k inferences sequentially, threading the recurrent state locally
   between them (state correctness requires sequential blocks; this also keeps one-job-in-flight). The result lambda pushes the
   k output blocks. To stay allocation-free: pre-warm in `resolveIO` (acquire a job, `reserve(K*C*N)` on input/state vectors,
   release it); note `JobPool<AudioInferJob>` is a process-wide static shared by all instances, so a job recycled from a
   smaller model can still grow on the audio thread — reserve to the max seen, or make the pool per-instance.
4. Output side: size `audio_out` for ≥ (K+1) blocks (currently one block + one host buffer, so a batched result or a late+early pair
   overflows and drops the oldest), and add a start-up prefill (don't pull until ≥1 block is buffered, or pre-seed one block of
   silence) so jitter becomes constant latency instead of gaps.
5. Related RT issues on the same path (pre-existing, not the cause of A1, same fix pass):
   - The result lambda captures `std::vector planar` and `new_states`, and `s.data = std::move(ns.second)` (`:510`) frees the old
     state buffer: all `free()` on the audio thread when the command is destroyed in `node_with_worker::run`. Fix: capture the
     pooled `std::unique_ptr<AudioInferJob>` (output + state buffers live in the job) in the lambda (8 bytes → fits the
     std::function SBO, so no heap node either), `swap` state buffers with the job's, and `JobPool::release` the job from the
     lambda (lock-free enqueue) instead of freeing.
   - `job->ishape = in_shape.tensorShape(n)` (`:282`) allocates a temporary vector every dispatch; fill it in place.
   - Reset during an in-flight job: the completion overwrites the freshly zeroed states (same as ledger V3). Carry a
     `generation` counter in the job; drop state (and optionally audio) from a stale generation.
   - Non-heavy models (< 32 MB and n ≤ 48000, `:265-266`) run `ctx->infer` synchronously on the audio thread, building
     `std::vector<Ort::Value>` per block (`:303`). Not RT-safe by design; out of A1 scope but worth a ledger line.
Risk / blast radius: AudioProcessor only (`dispatchInfer`, `operator()`, `worker::work`, `resolveIO`); `WaveformInput::prepare`
signature change touches AudioAnalyzer and TextToken callers (default the new parameter). Latency of async models increases by the
prefill (intended).
Test: port `a1sim2.cpp` into `tests/audio_nodes_test.cpp` (score-master has it untracked; the workshop tree has no tests/ audio
test): assert "dispatched ≥ 99% and zero consumed-and-dropped blocks" under a jittered latency < block period, and 100% with
batching when per-job overhead > block period.
Effort: M (gate + ring size: S; batching + RT-clean result lambda: M)

### A2 — 4-D stem outputs `[B,S,C,N]` are pushed as one mono stream
Status: CONFIRMED
Evidence:
- Demucs runtime output `(1, 4, 2, N)` (demucs.py). Declared `x[?,?,?,?]`, retagged Waveform by ModelArchetype.hpp:347-357.
- `AudioIO.hpp:259-263` default case of `fromInputShape`: `channels = 1; block = s.back()`.
- `AudioProcessor.cpp:476-478` (async; Demucs is 289 MB → async): `oc = os.channels > 0 ? os.channels : 1` → 1, `on = cnt/oc` = 8·N;
  `audio_out.push(planar, 1, 8N)`.
- `audio_out` was prepared with `out_shape.channels` = 1 and `model_block` = 1024 (`:191`) → ring cap ≈ 1024+512+4; `AudioRing::push`
  with `n >= cap` keeps only the last `cap` samples (AudioIO.hpp:154-160) → output is the tail of stem 3 / right channel, per block.
- Only one Waveform output exists, so `wave_out_indices.size() > 1` (`:255`) is false and Param 1 never selects anything.
- Host stereo out: `pull` maps channel 1 to `channels-1` = 0, so both host channels get the same scrambled mono.
Root cause: `WaveformShape` has no notion of a stem axis; stem selection only exists across separate output tensors.
Fix: add `int stems = 1` to `WaveformShape` and a `fromOutputShape(shape, fallback_channels)`: rank 4 `[B,S,C,N]` → `stems=S,
channels=C (1|2, else fallback), block=N`. In both decode paths select `s = clamp(int(param1*S),0,S-1)` and push
`planar + s*C*N` with `C` channels, `N` frames. In `resolveIO`, when the output channel dim is symbolic, prepare `audio_out` with
`in_shape.channels` (Demucs: 2) and with capacity for the real block (see A1.4). The stem count is only known at runtime (all dims
symbolic): do the selection from the runtime `osh`, never from the declared shape.
Also needed for Demucs to be usable at all: a block-size control (dynamic `l` currently defaults to 1024; Demucs costs ~11 s per call
regardless of length, so the block must be ~its segment, 343980 @ 44.1 kHz) — see A4 — and the rate fix (A3: Demucs is run at host rate).
Risk / blast radius: AudioProcessor decode (sync + worker); TextToken also uses `WaveformShape::fromInputShape` for its outputs
(TextToken.cpp:243) — keep `fromInputShape` unchanged and add the new function.
Test: unit test `fromOutputShape({1,4,2,-1})` → stems 4, ch 2; decode test that stem k channel c of a synthetic `[1,4,2,N]` buffer
lands in host channel c. ORT check: Demucs 343980-sample block, Param1=1.0 → vocals stem.
Effort: S–M

### A3 — model sample rate guessed from port names only
Status: CONFIRMED
Evidence: `AudioProcessor.cpp:47-70` matches only port names (`whisper, crepe, encodec, demucs, rave, deepfilter, dfnet, df_`),
else `return host_rate;`. Real port names: Demucs `mix`/`x` → host rate (should be 44.1 k); DTLN `input_2`… → host (should be 16 k);
hifigan `input`/`output` → host (22.05 k); deep-music-enhancer `x`/`y` → host. None of the tested files has rate metadata. The
comment "we can't read SR from the graph" ignores that `inputs.model.file.filename` is available (used at `:96`).
Root cause: port names almost never contain the family name; the file name (which does, e.g. `demucs__…`, `dtln__…`,
`silero-vad…`, `…clap…`) is not consulted, and there is no user override.
Fix: shared helper (e.g. `Onnx/helpers/AudioIO.hpp` or a new `AudioRate.hpp`) `double guessModelRate(const ModelArchetype&,
std::string_view filename, const ModelSpec&, double fallback)`: 1) `metadata_props` key `sample_rate`/`sr`/`sampling_rate` if
present (needs `ModelSpec` to read `GetModelMetadata()->LookupCustomMetadataMapAllocated`); 2) port names; 3) file name
(`demucs`→44100, `dtln|silero|crepe|whisper|yamnet|wav2vec|hubert`→16000, `hifigan|tacotron|vits`→22050, `encodec|vocos`→24000,
`clap|rave|deepfilter`→48000); 4) fallback. Plus a "Model rate" enum input (Auto / 8k / 16k / 22.05k / 24k / 32k / 44.1k / 48k)
that overrides, and re-`prepare`s the adapters when changed. Use the same helper in AudioAnalyzer (fixes AA3).
Risk / blast radius: AudioProcessor, AudioAnalyzer (AA3), possibly TextToken (vocoder rate). Changing Auto results changes the
behaviour of existing presets that relied on host rate by accident (they were wrong).
Test: table test of the helper over the real filenames above.
Effort: S (helper + filename) / M (with metadata + control)

### A4 — no hop / overlap-add (frame-based streaming models)
Status: PARTIAL — the mechanism is real, but DTLN (the ledger's example) would still not work with hop+OLA alone.
Evidence: `AudioProcessor.cpp:189` `audio_in.prepare(..., block, /*hop*/ block, ...)`; `WaveformInput` supports `hop < block`
(AudioIO.hpp:308, 387-388) but `WaveformOutput::push` (`:441-457`) only appends, no OLA/windowing.
DTLN is a two-model pipeline: dtln1 takes the **magnitude spectrum** `[1,1,257]` (rFFT of a 512 frame) and outputs a mask
(measured range 0.25–0.82 on random input), dtln2 takes the IFFT of the masked frame `[1,1,512]`; the reference runs 512/128 hop
with OLA. Under AudioProcessor, dtln1's `input_2[1,1,257]` is classified Waveform (`s[1]==1 && s[2]>32`) and fed 257 raw
samples; its output (a mask) is played as audio. dtln2 alone gets raw non-overlapping 512 blocks.
Root cause: hop is hard-wired to block; no output OLA; and no spectral frontend (A5) / multi-model chaining for DTLN.
Fix: add `Block` (Auto/explicit) and `Hop` controls (or a `hop = block/4` option for fixed-block streaming models), and a
`WaveformOutput::pushOLA(planar, C, N, hop)` with an accumulation buffer (preallocated `block` per channel) that emits `hop`
samples per call; optional synthesis window. Document that DTLN needs STFT (A5) + two nodes, or ship a merged DTLN export.
Risk / blast radius: AudioProcessor + AudioIO (new function; existing `push` unchanged).
Test: OLA unit test (hop=block/4, Hann² normalised, identity model → reconstruction error < 1e-4 after latency).
Effort: M

### A5 — no mel / STFT frontend
Status: PARTIAL — no frontend exists (true), but vocoders do not receive "raw samples in a `[1,80,T]` tensor": they throw.
Evidence: hifigan `input[1,80,T]` classifies Sequence (rank 3, `s[1]=80`), falls to the wave-in fallback (`:121-133`);
`fromInputShape` rank 3 → `channels = (s[1]==2)?2:1` = 1, block −1 → 1024 → tensor `[1,1,1024]`. ORT:
`Got invalid dimensions for input: input … index: 1 Got: 1 Expected: 80`. The exception hits `catch(...)` at `:242-245` →
`current_model_invalid = true` (sticky, X3) — the node goes silent. `Stft`/`Istft` exist in AudioIO.hpp:485-626 but are unused.
Root cause: design gap; `WaveformShape` cannot express `[1,F,T]`.
Fix: optional frontend enum (None / Mel(n_mels from shape[1]) / STFT-mag / STFT-complex) using the existing `Stft` + a mel
filterbank; until then, detect `[1,F>2,T]` inputs in `resolveIO` and refuse with a clear message rather than throwing per block.
Risk / blast radius: AudioProcessor. A true vocoder path also needs mel params (sr, n_fft, hop, fmin/fmax) per model family.
Test: shape test `[1,80,-1]` → refused/handled; with frontend, hifigan on a mel of a sine produces non-silent output.
Effort: L (frontend) / S (refuse cleanly)

### A6 — `AudioTaskMode` unused, Params 2–4 do nothing
Status: CONFIRMED
Evidence: `grep AudioTaskMode|param2|param3|param4` in AudioProcessor.* → only the enum definition (hpp:59-65) and the
slider declarations (hpp:87-89); `param1` is read once (cpp:258, stem select across multiple outputs).
Root cause: scaffolding never wired.
Fix: remove `AudioTaskMode` and Params 2–4 (renaming Param 1 to "Stem" once A2 lands). Removing ports changes the saved port
layout — do it before presets ship, or keep them hidden.
Risk / blast radius: saved documents / presets referencing those inlets.
Test: build; presets load.
Effort: S

---

## Audio Analyzer

### AA1 — state inputs pair with the first same-shape output (h and c both read hn)
Status: CONFIRMED for AudioAnalyzer and AudioProcessor; SequenceProcessor has a different (also broken) variant; VideoProcessor
has the same first-match shape fallback.
Evidence (pairing.cpp over the real signatures, loops copied from the nodes):
```
silero v4-16k : AA pairs: h->hn c->hn   AP pairs: h->hn c->hn   SP pairs: h->hn c->cn
silero-512    : AA pairs: h->new_h c->new_h   AP: same   SP pairs: h->(none) c->(none)
dtln2         : AA/AP: input_5->tf_op_layer_stack_5      SP pairs: input_5->(none)
h,c -> cn,hn (hypothetical output order): AA/AP h->cn c->cn   SP h->cn c->hn
```
- AudioAnalyzer.cpp:121-133 and AudioProcessor.cpp:160-174: loop over outputs, `break` on first shape match, no "claimed" set.
- SequenceProcessor.cpp:185-194, 206-222: skips claimed, but `sameShape` is exact equality **including dynamic dims**, so
  `[2,1,64]` vs `[2,-1,64]` (sherpa silero) or `[1,2,128,2]` vs `[-1,2,128,2]` (dtln2) never pair → states tagged but not
  threaded (fed zeros as aux, i.e. the model runs stateless). Also order-dependent (h→cn when outputs are cn,hn).
- VideoProcessor.cpp:466-470: shape fallback `if(spec.outputs[j].shape == in_shape …) return j;` — comment says "not already
  taken" but nothing tracks taken outputs.
- `classifyModel` (ModelArchetype.hpp:272-290) already skips claimed outputs and knows `h→hn`, `r#i→r#o`, `*_in→*_out`, but in a
  single pass with `named || sameShape` per output in output order, so a shape match earlier in the output list beats the
  real name partner (and it does not know `new_h`), and it throws the pairing away (only retags the output).
- Effect (silero.py, ALSA "Front_*" speech clips resampled to 16 k, 512-sample blocks; both silero files give identical numbers):
```
correct            speech mean 0.548 p90 0.999 frac>0.5 0.54 | noise mean 0.042 p90 0.105
AA/AP (c<-h_out)   speech mean 0.373 p90 0.888 frac>0.5 0.37 | noise mean 0.205 p90 0.296
```
  Direction matches the ledger (speech down, noise up ~5x); I did not reproduce the ledger's "0.07 on the sherpa export" — with
  my signal the sherpa (k2-fsa) file behaves exactly like v4.
Root cause: per-node, copy-pasted, first-match shape pairing without claim tracking or name priority.
Fix — one shared helper, owned by `classifyModel`:
- Add `int state_pair = -1;` to `ArchPort` (input: paired output index; output: paired input index).
- In `classifyModel`, after the tagging loop (keep tagging as is, but stop pairing inside `retagPairedOutput`), run a joint
  two-pass pairing over all RecurrentState inputs:
  1. name pass (all inputs first, so a later input's name partner can't be stolen by an earlier shape match):
     `r#i→r#o`, `*_in→*_out`, `x→xn`, `x→new_x`, `x→x_new`, `x→x_out`, `x→next_x`, also `state_in→state_out`/`past_*→present_*`
     if wanted; only unclaimed outputs.
  2. shape pass for the rest: first **unclaimed**, non-Image output with `sameShape` (existing wildcard lambda, ≥2 concrete dims).
  Retag each claimed output RecurrentState and set both `state_pair`s.
- Expose `inline std::vector<int> stateOutputFor(const ModelArchetype&)` (or just read `arch.inputs[i].state_pair`).
- Nodes: AudioAnalyzer (`resolveIO` L120-133), AudioProcessor (L159-174; additionally treat `o == wave_out_index` as claimed —
  pass it in or check after), SequenceProcessor (L200-230, replacing exact `sameShape`), VideoProcessor (`pairOutput`) all take
  `sp.out_index = arch.inputs[i].state_pair`. SequenceProcessor's "release pairs until a data output survives" logic stays.
Risk / blast radius: `classifyModel` is used by every node and by `tests/classify_*` (score-master): retagging behaviour for
outputs changes only in ordering (claimed-name-first); NodeKind routing is unaffected unless a model relied on the old
steal. The helper must keep `stateful` semantics. Silero in SequenceProcessor (X4 routes it there) becomes actually stateful.
Test: extend `tests/audio_nodes_test.cpp` `test_classify` (score-master) with the four signatures above and assert
`state_pair` (h→hn, c→cn; h→new_h, c→new_c; input_5→tf_op_layer_stack_5; swapped-order case h→hn, c→cn); add a check to
`tests/classify_real_models.py`; ORT regression = silero.py numbers (speech ≈0.55, noise ≈0.04).
Effort: M (S per node once the helper exists)

### AA2 — no per-frame normalisation for CREPE
Status: CONFIRMED
Evidence (crepe-full, 1024 samples @16 k):
```
f0   amp  raw                         normalised (x-mean)/std
110  0.02 bin 246 543.6Hz conf 0.16   bin 108 110.4Hz conf 0.88
330  0.02 bin 246 543.6Hz conf 0.11   bin 203 330.8Hz conf 0.96
880  0.02 bin 246 543.6Hz conf 0.11   bin 288 883.1Hz conf 0.97
330  0.5  bin 203 330.8Hz conf 0.85   bin 203 330.8Hz conf 0.96
```
AudioAnalyzer feeds `staged` unmodified (`runBlock` :209). Quiet input always lands on bin 246 (543.6 Hz).
Root cause: CREPE was trained on mean/std-normalised frames; nothing normalises.
Fix: "Normalize frame" toggle (or Auto when the input is `[?,1024]` and output `[?,360]`): per channel, subtract mean, divide by
`max(std, 1e-8)` on `staged` in place before building the tensor (RT-safe, no allocation). Also consider exposing Hz for the
CREPE case (`value1` is currently `argmax/(size-1)`, :329-330, not a frequency) and a hop < block (CREPE is normally 10 ms hop).
Risk / blast radius: AudioAnalyzer only; off by default except Auto-CREPE.
Test: the table above as an ORT check; unit test that the normaliser yields mean 0 / std 1.
Effort: S

### AA3 — analyzer rate: "clap" → 48 k, everything else 16 k
Status: CONFIRMED (and worse: the only CLAP export in the model set is not detected either)
Evidence: AudioAnalyzer.cpp:39-52 checks **input** names only for "clap"; the CLAP file's inputs are `longer`, `mel_fusion` → 16 k.
Silero and CREPE are correct at 16 k by default.
Root cause: same as A3 (names only, file name ignored, no override).
Fix: same shared helper and "Model rate" control as A3.
Risk / blast radius: AudioAnalyzer; CLAP cannot run anyway until AA4.
Test: same table test as A3.
Effort: S (with A3)

### AA4 — BOOL aux input fed float; rank-4 spectrogram input becomes `[1,1,64]`
Status: CONFIRMED
Evidence:
- AudioAnalyzer.cpp:235-244: only `Int64` gets a typed buffer; everything else (incl. `Bool`) → `vec_to_tensor<float>`.
  ORT: `Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bool))`.
- `mel_fusion[1,4,1001,64]` is the wave input (Spectrogram, :88), `fromInputShape` rank 4 → default case (AudioIO.hpp:259-263)
  → `tensorShape(64)` = `[1,1,64]` (pairing.cpp output). ORT: `Invalid rank for input: mel_fusion Got: 3 Expected: 4`.
- Either error lands in `catch(...)` (:185-188) → `current_model_invalid = true`, node silently dead (sticky, X3).
- Side note: `sr` declared `[]` is fed shape `[1]` (:231-232 `if(sh.empty()) sh = {1};`) — ORT accepts it for silero (checked), so
  not a bug today.
Root cause: aux feeding is float-or-int64 only; `WaveformShape` cannot represent 4-D spectrogram frontends, and nothing refuses them.
Fix: (1) fill aux by dtype: switch on `spec.inputs[i].elem_type` — Bool → `std::vector<uint8_t>`-backed `bool` tensor (value
`param > 0.5`), Int32/Int64 → integer (rate for `sr`-named, else `round(param)`), Float16 → converted; keep the per-input backing
vectors preallocated in `resolveIO` (currently they `assign` per block). (2) In `resolveIO`, if the wave input is rank 4 or a
Spectrogram with no frontend, set a clear "unsupported frontend (needs mel [1,4,1001,64])" state and skip inference instead of
throwing every block. A CLAP mel-fusion frontend is a separate feature (A5).
Risk / blast radius: AudioAnalyzer; the same dtype-blind aux feeding exists in AudioProcessor (`:329-338`, `:443-454`, always float
zeros) and SequenceProcessor (S2) — a shared `fillAuxInput(port, param, rate)` helper would fix all three.
Test: CLAP signature → no exception, node reports unsupported; a synthetic model with a bool input runs.
Effort: S (dtype + refuse) / L (CLAP frontend)


## Diagnosis — group `seq-geo-gen` (Sequence / Geometry / Image Generator)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx`. The sources of these three nodes and of `DataIO.hpp`, `GeometryIO.hpp` and `ModelArchetype.hpp` are byte-identical to score-master's (checked with `cmp`). Score was not launched, and nothing in the tree was modified.

Scratch: `docs/diagnosis-repros/seq-geo-gen/`:
- `io.py`: I/O dump.
- `mk.py` + `classify_cli`: the real C++ `classifyModel`, built from master's `tests/classify_cli.cpp` against the workshop headers.
- `s2s5.py`: Silero and Informer.
- `g1.cpp`: the real `classifyGeomOutput`.
- `geom_2in.onnx`: synthetic model for G2.
- `ig.py`: IG1–IG3.
- `montage.png` and the individual PNGs.

Classifier output for the test models (`classify_cli`):
```
silero-vad            in=Latent,Scalar,RecurrentState,RecurrentState out=Vector,RecurrentState,RecurrentState
informer_ETTh1        in=Sequence x4  out=Sequence
pointnet (both)       in=PointSet     out=Sequence,PointSet   <- trans[1,3,3] tagged PointSet
e4e decoder           in=Sequence([1,18,512]) out=Image
EigenGAN              in=Latent x7    out=Image
```

---

### S1 — Changing Window at runtime does nothing
Status: CONFIRMED (code)

Evidence:
- `SequenceProcessor.cpp:275-291`: `resolvedWindow` is computed from `inputs.window_mode.value` inside `reloadModel()` only.
- `:308-309`: `if(!ctx || lastModelPath != inputs.model.file.filename) reloadModel();` is the only trigger.
- No other read of `window_mode`.

Root cause: the window decision is cached at model load, and nothing compares the current enum with the one used to compute it.

Fix: in `SequenceProcessor.cpp`:
- Factor lines 275-291 into `void resolveWindow()`.
- Add a member `SeqWindowMode lastWindowMode`.
- In `operator()`, after the reload check: `if(inputs.window_mode.value != lastWindowMode) { resolveWindow(); window.configure(0,0); }`, with `resolveWindow()` storing `lastWindowMode`.

Risk / blast radius: this node only. It resets the ring on a change, which is the expected behaviour.

Test: a standalone test that the mode flips Passthrough↔Sliding without a reload. The node logic is not testable without ORT, so extend master's `tests/sequence_node_test.cpp` with a `resolveWindow(declared, mode)` pure function if it is factored out as a free function.

Effort: S

### S2 — Non-primary, non-state inputs are zero-filled (Silero `sr` = 0)
Status: CONFIRMED (repro)

Evidence:
- `SequenceProcessor.cpp:383-401`: "Any input the node didn't fill … gets a zero buffer". `filler[i].assign(flatPos(shp), 0.f)` → `buildTensor(..., spec.inputs[i].elem_type ...)`, so Silero `sr` (int64, rank 0) is fed as `0`.
- Repro (`s2s5.py`): 1 s of a modulated 220 Hz tone plus noise, 512-sample blocks, state threaded.
  ```
  silero sr= 0      mean prob 0.0042
  silero sr= 8000   mean prob 0.0042
  silero sr= 16000  mean prob 0.7865
  ```
  `sr=0` takes the 8 kHz branch, so the node's VAD output is essentially always about 0 on 16 kHz input.
- Informer's three aux inputs (`batch_x_mark`, `dec_inp`, `batch_y_mark`) also get zeros; see S5.

Root cause: there is no aux-input policy. Every unclaimed input gets zeros regardless of its name or role. Param 1/2 exist but are unused (S3).

Fix: use the shared aux filler (design at the end):
- `sr` / `sample_rate` (int) → model rate, default 16000.
- Scalar float inputs → Param 1, then Param 2.
- Anything else → zeros, as today.

Minimal local fix: in `runInference`, if `spec.inputs[i].name` is `sr`/`sample_rate` and the dtype is int, fill with 16000. Params have to cross into the job, so add `float params[2]` to `SeqInferJob`.

Risk / blast radius: this node's `runInference` only (sync and async share it). Models that currently rely on a zero aux value would change only if they match a name rule.

Test:
- `s2s5.py`-style check: Silero through the node's fill logic gives a speech probability above 0.5.
- Add a unit test of the helper's name/dtype rules.

Effort: S (local) / M (shared helper)

### S3 — Param 1/2 declared but never read
Status: CONFIRMED (code)

Evidence:
- `SequenceProcessor.hpp:89-90` declares `param1`, `param2`.
- `grep param OnnxModels/SequenceProcessor.cpp` finds no match.

Root cause: the controls were declared for aux scalars, but the wiring was never written.

Fix: same as S2. Bind them in the aux filler: the first two `Scalar`-archetype float inputs that no name rule claimed get Param 1 and Param 2. Carry the values in `SeqInferJob`.

Risk / blast radius: none beyond S2.

Test: a unit test of the helper. The Silero case does not exercise the Params, so add a synthetic 2-input model (data + scalar `t`, output = data·t) and assert the output scales with Param 1.

Effort: S (together with S2)

### S4 — No L2-normalise option for embedding heads
Status: CONFIRMED (repro)

Evidence:
- The raw head `wild2/nsfw_detector__clip-based-nsfw-detector__clip_nsfw_b32.onnx` (`input_1 [?,512] double`) was fed random 512-d vectors with norm 10 (a typical un-normalised CLIP image embedding), and then the same vectors L2-normalised:
  ```
  raw(norm10)=1.0000  l2normed=0.0021
  raw(norm10)=0.0000  l2normed=0.0000
  raw(norm10)=1.0000  l2normed=0.0876
  ```
  The un-normalised input saturates the head.
- The pack's `clip-vit-b32-nsfw-head-l2norm.onnx` starts with `LpNormalization` and gives the l2normed numbers for both inputs.

Root cause: `buildInput` passes the payload through as-is, with no per-vector normalisation stage.

Fix:
- Add an enum input `Normalize {None, L2, ZScore}` to `SequenceProcessor`.
- Apply it in `operator()` to `input` after `buildInput` (per frame of F for `[1,T,F]`, whole vector for `[1,D]`), before `dispatchInfer`.
- About 15 lines. ZScore also covers AA2-style per-frame normalisation if it is shared later.

Risk / blast radius: new control, default None, so no behaviour change.

Test: the raw head through the node with L2 should match the wrapper within 1e-4.

Effort: S

### S5 — Batch-baked exports fail inside infer(); the node dies without a usable message
Status: PARTIAL. The failure is confirmed. It is not fully silent (ORT's message goes to stderr), but the user sees nothing and the node is disabled for good.

Evidence:
- `s2s5.py` on `informer2020__informer_ETTh1.onnx`: the declared batch is **dynamic** (`batch_size`), but the graph bakes 2:
  ```
  informer B=1 ERR ... MatMul node '/decoder/layers.0/self_attention/out_projection/MatMul' ... dimension mismatch
  informer B=2 ok (2, 24, 7)
  ```
- `DataIO.hpp:94`: `r.shape[0] = 1; // batch always 1` also overrides a *concrete* declared batch greater than 1.
- `SequenceProcessor.cpp:394-397`: the filler also turns every dynamic dim into 1.
- `OnnxContext.hpp:403`: `fprintf(stderr, "ERROR running model inference: %s\n")` then rethrow.
- Sync path: informer is 45 MB (< 64 MB) with a count of 672, so it is not heavy. The exception reaches `SequenceProcessor.cpp:338-341` `catch(...) { current_model_invalid = true; }`, which is sticky (X3), so the node goes dark with no UI indication.
- Async path (`:541-544`): `catch(...)` just clears `inferenceInProgress`, so a large model would re-fail and re-log every tick.

Root cause: the batch is forced to 1 unconditionally, and graphs whose batch is baked while the declared dim is dynamic cannot be detected from the signature. Errors are not recorded anywhere user-visible, and the sync and async paths handle them differently.

Fix:
1. (S) At `reloadModel()`, run one probe inference with zero inputs at the resolved shapes.
   - On failure, store the message (`std::string load_error`), print it once with the model name, and set `current_model_invalid`.
   - Make the async catch set a flag that does the same, so it does not retry forever.
2. (M, optional) If the B=1 probe fails and the declared batch is dynamic, retry the probe at B=2.
   - If that succeeds, set `batch_override=2`: replicate the primary and aux payloads across the batch and decode batch slice 0.
   - Also honour a concrete declared batch > 1 in `resolveInputShape` the same way.

Risk / blast radius:
- A probe at load costs one inference. Skip it, or make it async, for "heavy" models.
- `resolveInputShape` / `buildInput` are used only by `SequenceProcessor.cpp` (grepped `Onnx/` and `OnnxModels/`).

Test: informer loads with a clear error (step 1), or produces `[24,7]` on Out after 96 frames (step 2).

Effort: S (1) / M (2)

---

### G1 — Task=Auto routes per-point seg logits (and the 3×3 transform) to the point cloud
Status: CONFIRMED (repro)

Evidence:
- The PointNet airplane model (`wild2/pointnet__airplane_100.onnx`, identical I/O to the preset's `pointnet-airplane-partseg.onnx`) outputs `pred (1,2048,4)`, values in [-61.7, 0] (log-softmax), and `trans (1,3,3)`.
- The real `classifyGeomOutput` (`g1.cpp`):
  ```
  [1,2048,4]  -> PointCloud layout=NPC C=4 N=2048
  [1,2048,13] -> PointCloud
  [1,3,3]     -> PointCloud  (N=3)
  [1,2048,50] -> Data        (ShapeNet-50 is only saved by the <=16 cap)
  ```
  `GeometryIO.hpp:429-434`: rank-3 plus `detectPointLayout().valid()` gives PointCloud. `:150-154` (`if(c2) setNPC`) accepts any last dim from 3 to 16.
- With Auto on output 0, Out gets 2048 "points" whose xyz are the first 3 log-probs, and Data stays empty.
- The shipped preset sets `Task="Data"` explicitly (`PointNet Airplane Part Segmentation.scp`, slot 5), which hides the bug for the preset only.

Root cause: the output classifier reuses the *input* layout detector, which deliberately accepts xyz+features (C from 3 to 16). For outputs, a channel count other than 3 almost always means per-point features or logits, not coordinates.

Fix: `classifyGeomOutput(oshape, in_layout, in_points)` in `GeometryIO.hpp`:
- PointCloud only if the layout is valid **and** `channels == 3` (or equals the input's channel count) **and** not `N==3 && C==3` (a transform matrix).
- Otherwise Data.
- Optionally, a name hint (`pred|logit|seg|score|cls|trans` → Data), since `spec.outputs[i].name` is available.

Pass `in_layout` and the supplied N from `decodeOutput` (both paths; the async job already carries `in_layout`).

Risk / blast radius:
- `classifyGeomOutput` is used only by `GeometryProcessor.cpp` (grepped).
- A completion model outputting xyz+normals `[1,N,6]` would move to Data under Auto. That is acceptable, and the Task override remains available.

Test:
- Extend master's `tests/geometry_node_test.cpp` with the shapes above: expect Data for `[1,2048,4]` and `[1,3,3]`, PointCloud for `[1,2048,3]` and `[1,3,2048]`.
- With the preset switched to Auto, Data should carry 2048×4 values.

Effort: S

### G2 — Only input 0 is bound; Param 1/2 never forwarded
Status: CONFIRMED (code + synthetic repro)

Evidence:
- `GeometryProcessor.cpp:334-340`: "Single-input binding … bound to declared input name[0]"; `in_dt = spec.inputs[0].elem_type`.
- `:367` / `:405`: `Ort::Value ins[1]`.
- `OnnxContext.hpp:389-396` passes `input_tensors.size()` (1) with `input_names_char.data()`, so ORT sees only name[0].
- `grep -c param GeometryProcessor.cpp` → 0, although `hpp:104-107` says the Params are "forwarded as extra scalar model inputs".
- Synthetic `geom_2in.onnx` (point `[1,N,3]` + `t [1]`), fed the point only: `Required inputs (['t']) are missing`. The C++ path was not run, but ORT's `Run` rejects missing inputs the same way.
- Second defect: `findCloudInput` (`:64-70`) may return an index other than 0, but the packed cloud is still bound to name[0] with input 0's dtype. A model with its cloud at index 1 is fed the cloud under the wrong name.

Root cause: the single-input binding is hard-coded.

Fix:
- Store `cloud_in` from `findCloudInput` in `reloadModel`.
- Build `std::vector<Ort::Value> ins(nin)`, putting the packed cloud at `cloud_in` with `spec.inputs[cloud_in].elem_type`.
- Fill the rest through the shared aux filler (Scalar → Param 1/2, rest zeros).
- Add `params[2]` to `GeomInferJob` for the async path.

Risk / blast radius: this node only. Single-input models are unchanged.

Test: in `geom_2in.onnx`, Out should equal the input + Param 1. Add it to master's `tests/geometry_node_test.cpp`, or as a Python-generated fixture.

Effort: S–M

---

### IG1 — Only input 0 is fed; multi-input generators (EigenGAN: 7) fail with no warning
Status: CONFIRMED (repro)

Evidence:
- `ImageGenerator.cpp:175-191` (`runStage`): `Ort::Value ins[1]`.
- `ig.py`, EigenGAN with only `eps` fed: `Required inputs (['z_', 'z_1', 'z_2', 'z_3', 'z_4', 'z_5']) are missing`.
- The worker catch (`:420-423`) only clears `inferenceInProgress`.
- `runGenerate` already set `produced = true; last_z = z_vector;` (`:327-328`), so the failure is not even retried until the latent changes. The only trace is the ORT stderr line.
- `reloadModel` (`:228-257`) never inspects the input count.
- Feeding all 7 inputs works:
  - All N(0,1): output `[1,256,256,3]` in [-0.98, 1.0], a valid anime face (montage, panels 6-7).
  - `z_*` = 0 also gives a valid image, with a mean abs difference of 0.42 against random `z_*`, so the extra inputs matter.

Root cause: the two-stage chain design assumes exactly one latent input per stage.

Fix:
- In `reloadModel`, classify the inputs of the *first* stage (and of the synthesis stage when chained) and build a per-input plan:
  - The primary latent is input 0, as today.
  - Every other float `Latent`/`Vector` input gets seeded N(0,1) from `seed + k` (k = input index), times Scale.
  - Scalars use Param 1/2 through the shared aux filler.
- `runStage` takes the full `ins` vector.
- Report a load message if an input can't be planned (e.g. int or image inputs).
- Optionally overlay the user Latent across the concatenation of all latent inputs, so EigenGAN's `z_*` eigen-dims become controllable.

Risk / blast radius: `runStage` and `GenJob` (it would carry the extra buffers and shapes). Single-input generators are unchanged.

Test: EigenGAN through `ig.py`-equivalent C++ gives a non-empty 256×256 image. Extend master's `tests/image_generator_test.cpp` with a plan test on the 7-input signature.

Effort: M

### IG2 — No w→w+ broadcast: mapping `[1,512]` can't chain into a w+ synthesis `[1,18,512]`
Status: CONFIRMED (repro). The fix was validated numerically and by eye.

Evidence:
- `ImageGenerator.cpp:221-226`: `chainCompatible` requires `flatNonBatch(map_out) == flatNonBatch(synth_in)` (512 ≠ 9216).
- `runGenerate` `:315-321`: an incompatible chain does a silent `return` every tick.
- Feeding `[1,512]` directly: `Invalid rank for input: latent Got: 2 Expected: 3`.
- With mapping `mobilestylegan-ffhq-mapping.onnx` (sequence-processor/) → `np.repeat(w,18,axis=1)` → `e4e-ffhq-decoder-wplus.onnx`, the result is a clean FFHQ face (montage panels 1-2).

Extra finding: the shipped **"e4e FFHQ decoder (w+)" preset has no mapping model**. It feeds iid N(0,1) straight into w+ space, which produces an unrecognisable blob (montage panel 3). The preset is broken until IG2 is fixed and the preset references the mapping net.

Root cause: the generic chain only accepts equal flat sizes, and StyleGAN w+ needs `w` tiled K times.

Fix:
- `chainCompatible`: also return true when `synth_in` is `[B,K,D]` with K>1, D == `flatNonBatch(map_out)`.
- In `worker::work` after stage 1: if `wcount * K == flatNonBatch(synth_in_shape)`, tile `synth_in` K times (`synth_in.resize(K*wcount)` plus copies) and keep the declared `[1,K,D]` shape. This must happen before the existing fallback at `:398-400`, which would otherwise flatten to `[1,wcount]`.
- Optionally, truncation `w = w_avg + psi·(w − w_avg)` via Param 1; that needs `w_avg`, so skip it for now.
- Update the e4e preset to set the Mapping model.

Risk / blast radius: `chainCompatible` is a public static used by the test; the stage-1 → stage-2 path is otherwise unchanged.

Test:
- Extend `tests/image_generator_test.cpp`: `chainCompatible({1,512},{1,18,512}) == true`, and `({1,512},{1,17,500}) == false`.
- Visual check as in `ig.py`.

Effort: S

### IG3 — Output Mode Auto = MinMaxNormalize; Denormalize is right for [-1,1] generators
Status: PARTIAL. The mapping is confirmed as described, but the visible impact is moderate, not catastrophic.

Evidence:
- `ImageGenerator.cpp:49-62`: `case Auto: return MinMaxNormalize`, with the comment "Dynamic-range GANs (MobileStyleGAN/EigenGAN)".
- Measured output ranges:

  | Model | Range | Mean abs pixel diff, MinMax vs Denorm | Mean luminance, Denorm → MinMax |
  |---|---|---|---|
  | StyleGAN2-1024 | [-1.06, 1.27] | 11.8 | 123 → 113 |
  | e4e (chained) | [-1.01, 1.33] | 23.0 | 164 → 141 |
  | EigenGAN (tanh, [-0.98, 1.0]) | [-0.98, 1.0] | 0.5 | — |

- So even EigenGAN, cited as the reason for MinMax, is a [-1,1] model.
- MinMax also rescales every frame by that frame's outlier pixels, so animating the latent pumps brightness. This is inferred from the per-frame min/max; flicker was not measured.
- The two shipped presets set Denormalize explicitly.

Root cause: the Auto default was chosen for a hypothetical dynamic-range GAN rather than the common tanh-output case.

Fix:
- Resolve Auto in `decodeOutput` (worker thread; the tensor is already in hand) from the actual range:
  - min ≥ −1.5 and max ≤ 1.5 with min < 0 → Denormalize.
  - min ≥ 0 and max ≤ 1.05 → DirectClamp.
  - max > 2 and min ≥ 0 → Passthrough (0-255).
  - Otherwise MinMax.
- Lock the choice on the first decode per model, so it doesn't flip frame to frame.
- Needs a `GenOutputMode` / flag in `GenJob` rather than a pre-resolved `WriteMode`.

Risk / blast radius: `resolveWriteMode` and `decodeOutput` in this node only. The same idea may apply to V2 (Video Processor).

Test: StyleGAN2 and EigenGAN resolve to Denormalize, and a synthetic `[0,1]` output resolves to DirectClamp (unit test on a pure `pickAutoMode(min,max)` function).

Effort: S

---

## Side notes (not ledger items, found while checking)

- **Order-dependent state pairing in Sequence Processor.**
  - `SequenceProcessor.cpp:205-219` pairs state inputs with the first unclaimed output of the **same shape**.
  - For Silero this is right only because the inputs `h,c` and the outputs `hn,cn` are in the same order.
  - A model declaring `c,h` / `hn,cn` would cross-wire them. This is the sibling of AA1.
  - Use the `classifyModel` name pairing (h→hn, c→cn, `*_in`→`*_out`) first, and fall back to shape.
- **The e4e preset is unusable as shipped** (see IG2).
- **`TokenIO.hpp::classifyAux` matches `"rate"` as LengthScale**, so an input named `sample_rate` would be mis-tagged. This matters if that function is reused for the shared filler below.
- **AudioAnalyzer builds `sr` as `{1}` instead of a rank-0 tensor.** ORT accepts it for Silero (tested: shapes `()` and `(1,)` both run). ORT rejects a **float** `sr`: `Unexpected input data type… expected tensor(int64)`. So dtype must be honoured, and rank is lenient but should be preserved.
- **`ImageProcessor` binds Scalar params as float always** (`ImageProcessor.cpp:703-712`). An int64 scalar input there would fail.

---

## Proposed shared helper: `Onnx/helpers/AuxInputs.hpp` ("aux input filler")

Today every node has its own "fill the inputs I don't own" loop:
- Sequence: zeros, dtype-aware.
- AudioProcessor: zeros, float only.
- AudioAnalyzer: int64 gets the rate, everything else gets `param`, float.
- ImageProcessor: Scalar gets Param 1/2 as float, the rest zeros as float.
- TextToken: its own `classifyAux` / `TokenAuxPlan`.
- Geometry and ImageGenerator: none.

S2, S3, G2, IG1, AA4 and T6 are all instances of this. The proposal is one dependency-free header, in the style of DataIO and GeometryIO: pure classification plus filling into typed caller-owned buffers, and a 10-line ORT wrapper per node.

```cpp
namespace Onnx {
enum class AuxFill : uint8_t {
  Zeros,        // default
  Ones,         // attention_mask / *_mask / valid
  False,        // bool flags (CLAP `longer`)
  SampleRate,   // sr / sample_rate / sampling_rate / fs  (int or float)
  Param,        // small float/int scalar -> Param k (param_index)
  SeededNoise,  // extra float latent (EigenGAN z_*, eps, noise*) -> N(0,1)*scale, seed+index
  LinkedLength, // *_length(s)/seq_len -> length of the primary input
  Refuse,       // past_* / cache / present / image-like / unknown rank>=3 int -> load error
};
struct AuxPlan {
  int index; TensorElemType dt; std::vector<int64_t> shape; // concrete, rank preserved (rank 0 stays [])
  AuxFill fill; int param_index = -1; float default_value = 0.f;
  int link_axis = -1; // for Ones/LinkedLength: which primary axis to mirror
};
struct AuxHost {            // what the node can supply at run time
  const float* params; int nparams;
  int64_t sample_rate;      // model rate (A3/AA3 table), 0 = unknown -> 16000 for "sr"
  uint32_t seed; float scale;
  const std::vector<int64_t>* primary_shape; // for mask/length linking
};
// Load time (once per model): first match wins, keyed on (lower-cased name, dtype, rank/size, archetype).
std::vector<AuxPlan> planAuxInputs(const ArchIO& io, std::span<const int> owned /*primary+state idx*/,
                                   int nparams, std::string* refusal_reason);
// Run time (RT-safe after first call): write values into typed per-input storage.
struct AuxStorage { std::vector<float> f32; std::vector<double> f64; std::vector<int64_t> i64;
                    std::vector<int32_t> i32; std::vector<uint8_t> u8; std::vector<uint16_t> f16; };
void fillAux(const AuxPlan&, const AuxHost&, AuxStorage&, std::vector<int64_t>& concrete_shape);
}
```

### Rule table
The table is ordered and data-driven, so each node can prepend overrides (e.g. the TextToken TTS roles).

| Name / type match | Dtype | Fill |
|---|---|---|
| `past_*`, `present*`, `*cache*` | any | Refuse |
| `sr`, `sample_rate`, `sampling_rate`, `fs` (whole-token match, **not** substring `rate`) | int/float | SampleRate |
| `*mask*`, `valid*` | int/bool/float | Ones, shape linked to primary |
| `*length*`, `*_len`, `seq_len` | int | LinkedLength |
| — | bool | False |
| float Latent/Vector with flat > 1 (generator nodes only, opt-in flag) | float | SeededNoise |
| Scalar archetype (`[]`, `[1]`, `[1,1]`, per I3) | float | Param k, in order |
| Scalar archetype | int | Param k, rounded |
| anything else | any | Zeros (today's behaviour) |

### Properties
- **Keyed on name and dtype.** The dtype decides the storage type, and is never a float fallback (fixes AA4 and the float-`sr` failure). Rank is preserved.
- **Dependency-free.** It sits beside `ModelArchetype.hpp`, is testable with the existing standalone test pattern, and `classifyModel`'s Scalar tag is reused.
- **Real-time safe.** Planning happens at `reloadModel`. At run time, `fillAux` only assigns into storage that is pre-sized once per model.
- **Async-safe.** A job carries `std::vector<AuxPlan>` (copied at dispatch; small) plus the `AuxHost` values, not pointers into the node.
- **Migration.** Sequence, Geometry and ImageGenerator adopt it first. AudioAnalyzer and AudioProcessor replace their loops (fixing AA4). TextToken keeps its TTS-specific roles as a prepended rule set.
- **Tests.**
  - One table test over real signatures: Silero `sr`, CLAP `longer`, BERT `attention_mask`, EigenGAN `z_*`, the synthetic geometry `t`, and informer `*_mark`.
  - Run it through `classify_cli`-style stdin fixtures, so `classify_real_models.py` can also sweep it.


## Diagnosis — group `texttoken` (T1–T9)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx` (the four relevant files are byte-identical to the score-master checkout).
score was not launched. Evidence comes from:
- static reading;
- a standalone clang++ repro against the real headers (`diag/texttoken/repro.cpp`, `st.cpp`);
- onnxruntime runs in Python (`diag/texttoken/rt.py`, `rt2.py`, `survey.py`, whose output is in `survey.txt`).

Scratch dir: `docs/diagnosis-repros/texttoken/`

The repro feeds dynamic dims as -1, as `readModelSpec` does.

**Important coupling.** T1, T2, T3 and T4 are all on the path for Piper. With T1 fixed alone, Piper still plays about 1 s, at the wrong pitch, re-triggering and chopped. Land them together.

---

### T1 — Piper `[B,T,1,N]` output is routed to Data, so no audio
Status: CONFIRMED

Evidence:
- `TokenIO.hpp:239-247` has rules only for rank 3, rank 2/1 and rank 1. There is no rank-4 rule, so the function falls through to `return TokenOutputRole::Vector` (L249).
- Declared output of `en_US-amy-low.onnx` (both the models-presets and the sherpa copy): `output FLOAT ['batch_size','time',1,'Unsqueezeoutput_dim_3']`.
- The ORT run returns shape `(1, 1, 1, 113152)`.
- The repro prints `piper amy-low … isTts=0 / out output Vector`.
- The file is 60 MB (over the 32 MB limit), so it runs async. The result goes to `outputs.data` (TextToken.cpp:574-580), and the audio outlet stays silent.

Root cause: `classifyTokenOutput` has no case for the standard Piper/VITS rank-4 output, which is `[B, T(=1), 1, N]`.

Fix: in `classifyTokenOutput`, before the rank-3 rule, add:
```cpp
if(rank == 4 && shape[2] == 1 && (shape[1] == 1 || shape[1] <= 0)
   && (last <= 0 || last > 32))
  return TokenOutputRole::Waveform;
```
Nothing else needs to change for rank 4:
- `WaveformShape::fromInputShape` already maps rank 4 through its `default` branch (channels=1, block=last=-1). That holds in resolveIO (L242) and at runtime (L445 / L583).
- `on = cnt` is correct for this shape.

Risk / blast radius: `isTtsModel()` uses this classifier. Rank-4 image-like outputs `[1,C,1,W]` with C>1 are excluded by the `shape[1]` guard. Nothing else calls it.

Test: add `CHECK(classifyTokenOutput("output",{-1,-1,1,-1},Float)==Waveform)` and a negative case `{1,3,1,64}` to `tests/texttoken_test.cpp`. That file exists only in score-master (untracked), not in the workshop tree, and it is not wired into CMake: the `score_addon_onnx_tests` target at CMakeLists.txt:676 does not list it.

Effort: S

---

### T2 — `guessTtsRate` always returns 22050
Status: CONFIRMED

Evidence:
- `TextToken.cpp:60-81` only greps port names for "16khz/44100/48000". Piper, Kitten and Kokoro port names contain none of these, so it returns 22050 (L80).

`metadata_props` checked with `onnx.load`:

| model | `sample_rate` | other relevant keys |
|---|---|---|
| Piper amy-low (both copies) | `'16000'` | `model_type:'vits'`, `comment:'piper'`, `n_speakers:'1'` |
| Kitten nano fp16 (both copies) | `'24000'` | `style_dim:'1,256'`, `n_speakers:'8'`, `speaker_names…` |
| Kokoro v0.19 | `'24000'` | `style_dim:'511,1,256'`, `n_speakers:'11'` |
| Matcha | `'22050'` | output is mel, not wave |
| Tacotron2 (wild), BERT NIDS, CLIP | none (empty) | |

- The Piper sidecar `en_US-amy-low.onnx.json` has `audio.sample_rate = 16000` and `inference = {noise_scale 0.667, length_scale 1, noise_w 0.8}`.
- There is no metadata lookup anywhere in `Onnx/` or `OnnxModels/`: grep for `Metadata` finds nothing relevant.
- Effect: Piper plays at 22050/16000 = 1.38x (too fast and high). Kitten/Kokoro play at 22050/24000 = 0.92x.

Root cause: the rate is never read from the model. It is guessed from port names that never carry it.

Fix (M, three layers):
1. In `reloadModel()`, read `ctx->session.GetModelMetadata().LookupCustomMetadataMapAllocated("sample_rate", alloc)`. `session` is a public member of `OnnxRunContext` (OnnxContext.hpp:291).
2. If that is missing, read the sidecar `<model filename>.json` → `audio.sample_rate`. This covers the upstream rhasspy Piper exports, which carry no `metadata_props`; the sherpa-repacked copy we have does.
3. As a last resort, add a "Model rate" override control, where 0 means auto. Keep the name-based guess and 22050 as final fallbacks.

Pass the result into `resolveIO` before `audio_out.prepare`. The sidecar's `inference` block could also seed the Param defaults. Optional.

Risk: only `TextToken::resolveIO` → `audio_out.prepare`. `model_rate` is snapshotted into the job but not otherwise used. `reloadModel` already runs on the DSP thread (it creates the session there), so one extra small file read does not worsen that pre-existing issue.

Test: in a unit test, load the Piper/Kitten sessions and assert 16000/24000. An integration test could check that output duration equals `samples/model_rate`: Piper, 250 ids → 113152 samples = 7.07 s at 16 kHz.

Effort: S–M

---

### T3 — TTS re-synthesises on every tick, so the output is chopped
Status: CONFIRMED

Evidence:
- `TextToken.cpp:300-301`: `const bool changed = !produced || ids != last_tokens; if(produces_audio || changed)`. For TTS the condition is always true.
- `dispatchInfer` (L330-331) only gates on `inferenceInProgress`. So as soon as a job returns, the next tick queues a new job with the same ids. Each result is pushed onto the ring (L590).
- Simulation in `repro.cpp`: 400 ticks × 512 frames (4.27 s @ 48 k), job latency 40 ticks, 1 s utterance at 16 k. Output: `dispatches=10, backward jumps (restarts)=24`. Playback restarts or overlaps continuously.
- Reset (L282-288) clears the ring, but the loop resumes on the next tick.

Related defect, same lines: for a heavy (>32 MB, async) text encoder such as CLIP 242 MB or ct-transformer 280 MB:
- `last_tokens = ids; produced = true;` are set before `dispatchInfer`.
- `dispatchInfer` then returns early if a job is in flight (L330).
- A change that arrives mid-inference is therefore lost for good, and the Data output stays stale.

Root cause: there is no trigger or state machine. For audio, "run when idle" is used as the trigger, and the change latch is committed even when the dispatch is dropped.

Fix — proposed state machine (per node; the DSP thread owns all state):
```
state:   gen (uint32), pending (bool), busy (bool = inferenceInProgress),
         last_ids, last_params[4]
inputs:  ids, params, Reset/Speak impulse

each tick:
  if Reset:  pending = true; player.stop()        // Reset == "speak again"
  if ids != last_ids || params != last_params:     // params: TTS only (optional:
       pending = true                              //  debounce ~100ms or only on ids)
  if pending && !busy && !ids.empty():
       last_ids = ids; last_params = params;       // commit ONLY on actual dispatch
       job.gen = ++gen; dispatch(job); busy = true; pending = false
  play/drain

on result (DSP thread, at start of tick):
  busy = false
  if job.gen != gen: drop                          // superseded while in flight
  else: player.start(result)                       // REPLACE policy: the new utterance
                                                   // interrupts the old one (clear+start);
                                                   // optional "Queue" mode appends instead
  (if pending is set, the next tick redispatches with the latest ids: latest-wins coalescing)
```
- `States`: Idle → (dirty) Pending → Busy → Playing → Idle (player exhausted, output zeros).
- A change while Busy only sets `pending`. There is at most one job in flight and at most one queued.
- The same latch (commit-on-dispatch) fixes the lost-change bug for async encoders.
- Consider adding an explicit `Speak` impulse and relabelling `Reset`. Without it, a user cannot re-speak the same ids.

Risk: `TextToken::operator()`, `dispatchInfer`, and the `work()` result lambda. It needs a `gen` field in `TokenInferJob`. Encoders keep the current behaviour (run on change) and additionally get latest-wins.

Test: a unit test of the state machine with a fake dispatcher, or drive `TextToken` with a stub `worker.request`. Assert:
- exactly one dispatch for constant ids over N ticks;
- one more after an ids change or Reset;
- a change during busy leads to exactly one follow-up dispatch carrying the latest ids.

Effort: M

---

### T4 — the ring is sized from the fallback 22050, so long utterances are truncated
Status: CONFIRMED. Nuance: the ring keeps the **tail**, so the beginning of the utterance is what is lost.

Evidence:
- `TextToken.cpp:247`: `const int64_t block = out_shape.block > 0 ? out_shape.block : 22050;`. Every TTS we have declares a dynamic N: Piper `-1`, Kitten `num_samples`, Kokoro `audio0`.
- `AudioIO.hpp:418-420`: `cap = ceil(block*ratio)+2 + max_host_frames + 2`.
- `AudioRing::push` (AudioIO.hpp:154-160): if `n >= cap`, it keeps only the most recent `cap` samples.
- Repro output (push a 3 s utterance, host 48 k, 512 frames):
  - `model_rate=22050 ring_cap=48517 (1.011s) … 1.989s lost` ← current behaviour, since T2 forces 22050
  - `model_rate=16000 ring_cap=66666 (1.389s) … 1.611s lost`
  - `model_rate=24000 ring_cap=44616 (0.929s) … 2.070s lost`
- Real utterance length for one two-sentence line (ORT):
  - Piper: 7.07 s
  - Kitten: 5.3–6.8 s
  - Kokoro: 7.6 s

Root cause: the realtime ring is a streaming FIFO sized for one model block. TTS emits one variable-length, multi-second block per job. Its size is known only in the worker, and the ring cannot grow on the DSP thread.

Fix (recommended, M): do not use the streaming ring for one-shot TTS.
- In `worker::work()` (worker thread), resample the whole utterance to host rate with a fresh `ResamplerLin` per channel. Utterances are independent, so no phase carry is needed.
- Return the host-rate planar buffer in the result lambda.
- On the DSP thread, a small `OneShotPlayer { std::vector<float> buf; size_t pos; int ch; }` receives it with a `swap`.
- The old buffer goes back into the lambda capture, so it is freed wherever the lambda dies, not in `pull()`.
- `pull` copies `min(frames, remaining)` and zero-fills the rest. No ring, no truncation, and no RT allocation.
- This is the "player" in the T3 state machine. Replace policy = swap; queue policy = keep a second slot.

Minimal alternative (S):
- `prepare` the ring and `rs_scratch` for a max utterance length (e.g. 30 s × host_rate: about 1.44 M floats, 5.8 MB per channel).
- In `push`, clamp `n` to what fits, keeping the head and dropping the tail rather than the reverse.
- This still truncates beyond the cap and costs memory per node.

Risk: `WaveformOutput` is shared with other nodes (AudioProcessor etc.), so do not change `AudioRing` semantics globally. Keep the change local to TextToken, or add a new type.

Test: push a 7 s buffer and assert every sample is drained in order across N pulls (the first sample equals the utterance start). Also, via the repro harness, assert no allocation happens in `pull`.

Effort: M (S for the minimal version)

---

### T5 — the token tensor is always int64, so int32-token models fail
Status: CONFIRMED (and broader than the ledger says: int aux inputs are also always int64)

Evidence:
- The token tensor is built as int64 on both paths:
  - sync: `vec_to_tensor<int64_t>(token_buf, …)` (TextToken.cpp:376);
  - async: L499-500.
- The int aux tensors are also int64 on both paths (L391-401 and L522-530).
- ORT results:
  - `punct int64 tokens … FAIL INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(int64)) , expected: (tensor(int32))`
  - `ct-transformer int64 … FAIL (same)`, while the int32 run is OK: `(1, 6, 6)` logits.
- Affected models with INT32 inputs:
  - sherpa `online-punct-en` (`token_ids`, `valid_ids`, `label_lens`);
  - `ct-transformer` (`inputs`, `text_lengths`);
  - moonshine `args_0` / `args_2`.
  - Tacotron2 `decoder_iter` has a BOOL `mask`. `isIntType` counts Bool as int, so it too would be fed int64.
- Failure mode depends on model size:
  - punct-en is 28 MB, so it runs sync. `infer` throws, `catch(...)` sets `current_model_invalid`, and the model is silently dead.
  - ct-transformer is 280 MB, so it runs async. The worker's `catch(...)` (L593-596) swallows the error. Nothing is surfaced, and it is retried on every change.

Root cause: the dtype is hardcoded to int64. The declared `elem_type` is read (L388, L519) but only used to choose int vs float.

Fix:
- Keep the int64 staging buffer.
- At tensor creation, switch on the declared dtype: Int64 → as is; Int32 → convert into a reusable `std::vector<int32_t>`; Bool/Uint8 → `std::vector<uint8_t>` (bool is 1 byte in ORT); fall back to int64 otherwise.
- Apply the same to the int aux inputs.
- Carry the token dtype in `TokenInferJob`. A small helper `makeIntTensor(dt, int64 src, shape, bufs)` shared by both paths removes the duplicated sync/async code.
- Also surface async errors: return a lambda that sets `current_model_invalid` or an error flag instead of swallowing them.

Risk: both input-building paths. The existing int64 models are unchanged.

Test: an ORT-backed test that builds the inputs for `online-punct-en/model.onnx` and runs them. Expected output: `(4,4)` case and punct logits, given valid_ids of ones (see T6).

Effort: S–M

---

### T6 — a dynamic aux int input (BERT `attention_mask[?,?]`) becomes `[1,1]` holding `round(Param1)`
Status: CONFIRMED (and also applies to int32 `valid_ids` in punct-en)

Evidence:
- `resolveShape` maps every dynamic dim to 1 (TextToken.cpp:47-55).
- `attention_mask` (INT64 `[batch_size, sequence_length]`) → repro `aux=GenericInt` → `takeParam()` → Param 1 (L215-216).
- Only `buf[0]` is set (sync L397-398, async L528-529). With the default 0.667 → `lround` = 1.
- ORT, BERT NIDS model:
  - `mask=[1,1] fill 1 (node) FAIL … Reshape_3 … Input shape:{1,1}, requested shape:{1,1,1,6}`
  - `mask=ones[1,L]` OK
  - `L=1, mask [1,1]` OK, so it works only for single-token inputs
  - `mask=[1,1] fill 0` FAIL
- punct-en with `[1,1]` `valid_ids`/`label_lens` runs, but returns **empty** `(0,4)` logits. Given `ones[1,L]` + `label_lens=[L]`, it returns `(4,4)`. The node's output is empty or wrong, with no error.

Root cause: aux inputs whose dynamic dims are the token-sequence dims are not tied to the token shape. A mask is treated as a generic scalar Param.

Fix: in `resolveIO`, when building aux plans:
- For each int aux input, if `rank == token rank` and it has dynamic dims at the token's L position, set `role = Mask`. The name hints `mask`, `valid`, `attention`, `token_type` help, and shape-equality with the token input is enough on its own.
- At dispatch, give it the token tensor shape. Fill it with 1 for `*mask*`/`valid*`, and 0 for `token_type_ids`/`segment`.
- Also: rank-1 dynamic int inputs named `*lens*`/`*len*` (`label_lens`) should be `InputLength`. The `classifyAux` name list lacks "lens"; it has "lengths" and "length".
- More generally, `GenericInt` / `GenericFloat` with a dynamic dim should never be `[1,…]` scalar-filled. Either map them to the token shape or refuse the model.

Risk: `classifyAux` and `resolveIO`. The TTS models are unaffected: their aux inputs are `input_lengths`, `scales[3]` and `speed[1]`.

Test: ORT test on the BERT NIDS model with L=6: expect `(1,24)` logits, and equal to the reference run with `ones`.

Effort: S–M

---

### T7 — `isAutoregressive` misses Tacotron2 `decoder_iter`, pocket-tts `lm_main` and moonshine `cached_decode`
Status: CONFIRMED

Evidence:
- The repro prints `isAutoregressive=0` for all three, and also for moonshine `uncached_decode`.
- The name list is at `TokenIO.hpp:155-161`: `past_*`, `present`, `cache*`, `position_id`, … It never matches:
  - `attention_hidden` / `attention_cell` / `decoder_*` / `memory`;
  - `state_0..17` / `out_state_*`;
  - `args_0..26` / `functional_*`.
- What `classifyModel` detects (`st.cpp`):

| model | `stateful` | isAutoreg | `tokenSeqIdx` |
|---|---|---|---|
| taco | 1 | 0 | 10 |
| pocket | 1 | 0 | -1 |
| moon_cached | 0 | 0 | 0 |
| moon_uncached | 0 | 0 | 0 |

- For Tacotron, the "token" chosen is the BOOL `mask [1,seq_len]` (index 10), because `isIntType(Bool)` is true.

Root cause: the refusal relies only on a KV-cache name list. It neither uses `arch.stateful` nor checks whether the node can feed the model's inputs at all.

Fix (two gates in `reloadModel`):
1. `refused = isAutoregressive(io) || arch.stateful;`. This catches Tacotron and pocket, which carry name-flagged or paired state.
2. Add an "unfeedable input" gate. Refuse if any non-token input is a float/int tensor with `rank >= 3`, or with a dynamic dim that is not tied to the token length (after the T6 mask handling). The node can only synthesise scalars, small vectors and masks.
   - This catches moonshine cached (`args_1 [.,.,288]`, `args_3.. [.,.,8,36]`), moonshine uncached (`args_1`), Tacotron (`memory`), and pocket (`state_*` rank 5).
   - Kitten/Kokoro `style [1,256]` stays allowed, as a T9 port.
3. Exclude `Bool` from token selection. Optionally add `"state_"`, `"hidden"`, `"_cell"` to `isAutoregName`, as cheap belt-and-braces.

Also surface the refusal: today it is a silent no-op (L277).

Risk: `isAutoregressive` is used only by TextToken. Check that CLIP, Piper, Kitten, Kokoro, BERT and punct are not refused: none of them have rank≥3 non-token inputs.

Test: extend `st.cpp` into a unit test with the four exported I/O tables. Expect refused=1 for all AR models and 0 for the six single-forward models.

Effort: S

---

### T8 — `WaveformOutput::push` `rs_scratch` grows (RT allocation); `TextTokenTask` is unused
Status: CONFIRMED

Evidence:
- `AudioIO.hpp:424-429` reserves `per_block + 4`, with `per_block` derived from the fallback 22050.
- `push` (L452-455) calls `rs_scratch[c].clear(); resamplers[c].process(…, rs_scratch[c])`, which does `push_back` with no bound (L91, L111).
- Repro: `rs_reserve=66156 → after 3s push: rs_cap=264624 (REALLOC=1)`, and the same for all rates.
- `push` runs on the DSP thread. Per `Crousti/Executor.hpp:65-75` and `:926-935`, the result lambda runs "back in the processing thread … at the beginning of the node's own tick".
- The result lambda also carries the `planar` vector, which holds the whole utterance: `std::vector<float> planar(f, f + cnt)` in `work()` L570. That is fine on the worker, but its lifetime ends with the lambda.
- On the sync path, `dispatchInfer` itself allocates on the DSP thread:
  - `ins`, `aux_*_bufs.assign`, `outs`, `planar` (L365-368, L427, L448);
  - it also runs ORT inline (L431).
  - This is by design for "light" encoders but still not RT-safe.
- `TextTokenTask` is declared at `TextToken.hpp:87-92`. grep finds no other reference in `OnnxModels/`, `Onnx/` or `tests/`, and there is no Task input port.

Root cause: the scratch is sized for one streaming block, but TTS pushes whole utterances. `TextTokenTask` is a leftover of an unimplemented override.

Fix:
- It is subsumed by the T4 recommended fix: resample in the worker, then a one-shot player with no `rs_scratch` on the DSP thread.
- If the ring is kept, bound `push` in chunks of `≤ per_block` input samples (a loop over `n`), so `rs_scratch` never exceeds its reserve.
- For `TextTokenTask`, either delete the enum, or add a `halp::enum_t<TextTokenTask,"Task">` input and apply it in `resolveIO` (Audio → force `produces_audio` with `wave_out_index=0`; Data → force false). The ledger lists this as a two-line cleanup.

Risk: the chunked `push` also touches the other users of `WaveformOutput` (a behaviour-preserving change).

Test: a repro that checks `rs_scratch[0].capacity()` is unchanged after pushing 10× `per_block`.

Effort: S

---

### T9 — style-vector TTS (Kitten, Kokoro) gets a constant Param as `style[1,256]`
Status: CONFIRMED (measured; not listened to)

Evidence:
- `style FLOAT [1,256]` → repro `aux=GenericFloat` → `takeParam()` → Param 1 (default 0.667), filled across all 256 values (sync L414-417, async L544-547).
- `speed` → `LengthScale` → Param 2.
- `voices.bin` layout, confirmed with numpy:
  - Kitten: 2048 floats = 8 voices × 256.
  - Kokoro: 1,438,976 floats = 11 voices × 511 × 256, i.e. `style_dim '511,1,256'`. The row is selected by token count.
- Same ids, ORT (wavs written to the scratch dir):

| style | Kitten | Kokoro |
|---|---|---|
| voice0 (real) | rms 0.112, peak 0.61, 6.75 s | rms 0.055, peak 0.40, 7.62 s |
| constant 0.667 (node default) | rms 0.012, peak 0.08, 5.25 s | rms 8.29, **peak 576**, 3.65 s |
| zeros | rms 0.066, peak 0.45, 5.72 s | rms 2.40, **peak 78**, 6.70 s |

- So Kokoro's output with any constant style is badly out of range: it clips or explodes. Kitten's output with the default is 10x too quiet and shorter.
- Intelligibility was not judged by ear. The files are `kitten_*.wav`, `kokoro_*.wav` and `piper.wav`.

Root cause: the node has no way to supply a speaker embedding. A 256-wide float input falls into the generic Param mapping.

Fix (M):
- Add a `halp::file_port<"Voices">` (raw float32 file), plus use Param 4 / speaker as the voice index.
- In `resolveIO`, tag a float input named `style`/`speaker_emb`/`spk_emb` with shape `[1,D]` as `AuxRole::Style`.
- Read the model metadata `style_dim` (`"1,256"` or `"511,1,256"`) and `n_speakers` to interpret the file:
  - voice v, row `r = (rows > 1) ? min(L, rows-1) : 0`;
  - copy 256 floats into the job.
- If there is no file, refuse or emit silence rather than garbage, or default to row 0 of voice 0 if a sibling `voices.bin` is found next to the model.
- Separately, `speed` is inverse to `length_scale` (higher is faster), while Param 2 is labelled "length". Document or invert this.

Risk: `classifyAux` and `resolveIO`, plus the job snapshot (256 floats; reserve it in the pooled job so it does not allocate). There is a new port, so the UI and presets change.

Test: an ORT test on Kitten with `voices.bin` voice 0. Assert the output rms is in [0.05, 0.3] and peak < 1. For Kokoro, check that the row is indexed by L.

Effort: M

---

## Extra findings (not in the ledger)

- **X-a:** a change to a heavy async encoder made during inference is lost. Covered by the T3 latch.
- **X-b:** the async `work()` `catch(...)` swallows every error (TextToken.cpp:593-596). The node gives no indication and retries forever: every tick for TTS, every change for encoders. Surface an error flag or set `current_model_invalid`.
- **X-c:** the token input for Tacotron `decoder_iter` is the BOOL mask, because Bool counts as int in `isIntType`. Covered by the T7 refusal; also exclude Bool from `TokenSeq`.
- **X-d:** `classifyAux` misses `lens` (punct `label_lens`), so it becomes GenericInt → Param. Fold this into T6.


## Diagnosis — group `llm-vlm` (L1, L2, F1, F2)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx` (read only). Binaries used: `build-developer/score_addon_onnx_{qwenllm,fastvlm}_smoke`, run with `SCORE_ONNX_FORCE_PROVIDER=cpu`.
Scratch artefacts are in `docs/diagnosis-repros/llm-vlm/`:
- `io.py`: dumps model I/O.
- `order.py`: checks the positional layout.
- `smol-permA/`, `smol-permB/`: permuted SmolLM2 copies.
- `lfm2_byname.py`: a prototype of by-name binding for LFM2.

---

### L1 — QwenLLM binds decoder inputs and outputs by position
Status: **CONFIRMED**. The mechanism is exactly as described. **PARTIAL** on the example: LFM2 is not silently mis-bound. It is refused at load with a misleading message.

Evidence:
- `Onnx/helpers/QwenLLM.cpp`:
  - 518-554 push the inputs in a fixed order: `input_ids, attention_mask, [position_ids], (k,v)×numLayers`.
  - 556-562 call `Run(inputNamePtrs.data(), inputs.data(), std::min(inputs.size(), inputNamePtrs.size()), …)`. This pairs the i-th value with the i-th graph input name. The `std::min` also silently drops trailing values or names.
  - 464 and 474: `outs[1 + layer*2]` and `outs[1 + layer*2 + 1]` assume the output order `logits, present.0.key, present.0.value, …`.
  - 146-177: the geometry comes from counting `past_key_values.*`, and `kvHeads`/`headDim` are read only from the literal name `past_key_values.0.key`.
- All 16 LLM exports on disk match the positional layout exactly (`order.py`): DeepSeek, Falcon3, Llama-3.2 1B/3B, Phi-4-mini, the Qwen2.5 variants, genai-fp16, Qwen3 0.6B/1.7B, SmolLM2 and gemma-3-1b. So nothing shipped is broken today.
- LFM2 (`/mnt/win2/models/LFM2-1.2B/onnx/model_q4.onnx`):
  - Inputs: `input_ids, attention_mask, num_logits_to_keep, past_conv.0, past_conv.1, past_key_values.2.key, …`. It has 10 conv layers of `[b,2048,3]` and 6 attention layers (2, 5, 8, 10, 12, 14).
  - Smoke output: `FAIL: QwenLLM: could not derive the KV cache geometry from the model (not a transformers.js-style decoder export?)`. The cause is that there is no `past_key_values.0.key`. The load is refused rather than mis-bound.
- The silent mis-bind is real. I proved it with permuted copies of SmolLM2-360M q4. The graph is unchanged; only the order of the `graph.input` / `graph.output` lists differs:
  - Original: `The capital of France is Paris.`
  - permA (the `attention_mask` and `position_ids` inputs swapped): `FAIL: … GroupQueryAttention … seqlens_k[0] = 779 is out of range`. This one is loud.
  - permB (the outputs `present.0.key` and `present.0.value` swapped): `TheThe capitalpital of Francepіtі of Francepіtі of Francepftt of.`, and the smoke reports **0 failures**. This one is silent garbage.

Root cause: the names are enumerated from the session, but values are assembled in a hard-coded order and results are indexed by a hard-coded offset. Any export whose inputs or outputs are in a different order, or that has extra inputs, is mis-fed. When dtype and rank coincide there is no error. Examples are key/value swapped, `position_ids`/`attention_mask` swapped, or outputs with `present.*` before `logits`.

Fix (bind by name). This is in scope and self-contained in `Onnx/helpers/QwenLLM.{hpp,cpp}`:
1. At load, classify every input name. Anything else throws, naming the offending inputs, so unknown inputs are refused explicitly.
   - `input_ids`, `attention_mask` and `position_ids`: fed from the per-step values, as today.
   - `num_logits_to_keep`: an int64 scalar, shape `{}`, value `1`. `readLastLogits` already takes the last row, so it works unchanged, and prefill no longer materialises `[1, promptLen, V]` logits.
   - `past_key_values.N.key|value`: output `present.N.key|value`. Initial shape `[1, dim1, 0, dim3]`.
   - `past_conv.N`: output `present_conv.N`. Initial shape `[1, dim1, dim2]` taken from the concrete input dims (LFM2: `[1,2048,3]`), **zero-filled**, and it does not grow.
   - Replace `keyCache/valueCache/cacheShapes[numLayers]` with a generic `std::vector<StateSlot>`, where each slot has `{inputIdx, outputName/outputIdx, dtype, shape, bytes}`. Take the dtype per slot from its own input type info; do not assume one global `kvType`. The initial KV shape comes from each slot's own dims, so `kvHeads/headDim` and the lookup of `past_key_values.0.key` go away.
2. Build `inputs` in session-input order: iterate `inputNames` and pick the value for each role. Drop the `std::min`, and `Run` with all names.
3. Request outputs by the explicit list `{"logits", slot.outputName…}`. Index the results from that list, not by `1+2l`. Unrequested outputs are skipped automatically.
4. Validate that every state input has a matching output name. Otherwise throw.

Prototype that validates the LFM2 part end to end, with Python ORT and a hand BPE tokenizer: `python3 lfm2_byname.py` prints
`logits shape (1, 1, 65536) conv state (1, 2048, 3)` and `REPLY: 'The capital of France is Paris.'`.
It uses zero-initialised `past_conv`, carries `present_conv.N` to `past_conv.N`, uses `num_logits_to_keep=1`, a full-length `attention_mask`, and stop id 7 from `generation_config.json`. `readStopTokens` already picks up that stop id.

Scope: yes. The generic state-slot approach covers both LFM2's conv state and `num_logits_to_keep` for about the same code as fixing the ordering. Not verified in C++:
- that `OrtxApplyChatTemplate` renders LFM2's `chat_template.jinja` (it has a `{{- bos_token -}}` prefix plus tools logic);
- that ortx tokenizes it identically. `OrtxCreateTokenizer` does succeed on LFM2, because the smoke got past it.

The same positional pattern exists in FastVLM; see the F1 note.

Risk / blast radius: only `QwenLLMInference`, the "Language Model" node. All current exports keep working provided the name classification covers `position_ids`. Watch the per-slot dtype: the genai fp16 export has fp16 logits and KV, and Qwen3 fp16 has fp32 logits with fp16 KV.
Test:
- Extend `tests/qwenllm_smoke.cpp`: LFM2 q4 must answer "Paris".
- Add a permuted-order model check, generating `smol-permB` with the ~10-line `onnx` script in this report's scratch dir. It must still answer "Paris".
- Add a synthetic model with an unknown extra input. It must throw a clear error.
Effort: **M**.

---

### L2 — `<think>` blocks (Qwen3, DeepSeek-R1) are emitted and count against Max tokens
Status: **CONFIRMED**.
Evidence:
- `qwenllm_smoke Qwen3-0.6B`: the response is `<think>\nOkay, so I need to answer the question: … Let me`. All 32 tokens are spent inside the think block and there is no answer.
- `qwenllm_smoke DeepSeek-R1-Distill-Qwen-1.5B q4`: `<think>\nOkay, so I need to figure out the capital of France. …`, same result.
- `<think>` and `</think>` are non-special added tokens: Qwen3 151667/151668, DeepSeek 151648/151649. Detokenisation therefore keeps them.
- `OnnxModels/QwenLLM.cpp`:
  - 204-212 forward every delta to `pending`;
  - 148 appends everything to `accumulated_response`;
  - 239 emits it as Response.
- `generateLoop` (`Onnx/helpers/QwenLLM.cpp:516`) counts every sampled token against `maxTokens`.
- Qwen3's template ends with `{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}`. `OrtxApplyChatTemplate(tokenizer, template_str, input, tools, out, add_gen, tokenize)` has no kwargs parameter, so `enable_thinking` cannot be passed.

Root cause: there is no reasoning-model awareness. Think tokens are ordinary text to the node.

Fix: a "Thinking" enum port appended after `partialMode`, to keep the inlet ids stable, with three values:
- **Off**: Qwen3-style. If the template source contains `enable_thinking`, append `"<think>\n\n</think>\n\n"` to the rendered prompt in `applyChatTemplate`. This is byte-identical to `enable_thinking=false`. It is not possible for R1-distill, which always thinks, so there fall back to Hide.
- **Hide**: in `generateLoop`, look up the ids of `<think>`/`</think>` once at load with `tokenize("<think>")`, and use them only if each yields a single id. Suppress `onToken` between them. Optionally keep a separate, bounded think budget so Max tokens applies to the visible answer, for example total cap = Max tokens + think budget.
- **Show**: today's behaviour.

Risk / blast radius: only the LLM node. A new port is appended, so presets are unaffected.
Test: smoke with Qwen3-0.6B Off, which should answer "Paris" within 32 tokens, and DeepSeek-R1 Hide, where the response must not contain `<think>`.
Effort: **S** (Off plus Hide), **M** with a separate think budget.

---

### F1 — FastVLM hard-codes the Qwen2 BOS/EOS/`<image>` ids and the ChatML template
Status: **CONFIRMED** for the hard-coding. **PARTIAL** on the impact: reading the ids alone would not make any other VLM work.

Evidence:
- The ledger's "FastVLM.cpp:32-35" is `Onnx/helpers/FastVLM.cpp`, not `OnnxModels/`:
  - 32 `BOS_TOKEN_ID = 151643`;
  - 33 `EOS_TOKEN_ID = 151645`;
  - 34-35 `IMAGE_TOKEN_INDEX = 151646`;
  - 39 `"<image>"`.
- Template at 547-555: `"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{}<|im_end|>\n<|im_start|>assistant\n"`.
- The ids are used at:
  - 489 and 585: image sentinel insertion and replacement;
  - 522: BOS;
  - 830 and 882: the single EOS check;
  - 298-310: string-stripping of `<|im_start|>assistant\n` / `<|im_end|>`.
- For FastVLM-0.5B the hard-coded values equal its config, and the smoke q4 run gives "The image displays a gradient of colors ranging from purple to yellow…".
- SmolVLM-256M q4 fails in the vision encoder: `Invalid rank for input: pixel_values Got: 4 Expected: 5`. Its inputs are `pixel_values[b,num_images,3,512,512]` and `pixel_attention_mask:BOOL`.
- gemma-3-4b-it q4 fails in the vision encoder: `/model/embeddings/Add … Attempting to broadcast`. Its input must be 896×896, and the helper feeds the native image size; see `preprocessImageForFastVLM` at 42-63, which does no resize.
- Gemma's decoder also has `num_logits_to_keep` as input 2 and no `position_ids`. The decoder is positional too:
  - 195 `numLayers = (nIn - 3) / 2`;
  - 715 and 871 use `std::min(...)` in Run;
  - 807 and 817 use `outs[1 + layer*2]`.

  So gemma would feed `position_ids` into `num_logits_to_keep`. This is the same bug class as L1.

Config keys that carry the ids:

| Model | image token id | stop / eos | bos | template | image-token expansion |
|---|---|---|---|---|---|
| FastVLM-0.5B (`llava_qwen2`) | `config.json` → `image_token_index` = 151646 (a sentinel ≥ vocab_size 151646, not in the tokenizer); `processor_config.json` → `image_token` = `"<image>"` | `config.json` / `generation_config.json` → `eos_token_id` = 151645; `tokenizer_config.json` → `eos_token` = `<|im_end|>` | `bos_token_id` = 151643 (`tokenizer_config.bos_token` = null, so no BOS is prepended) | `tokenizer_config.json` → `chat_template` (the same ChatML as the hard-coded one) | one sentinel, replaced by all N vision features |
| SmolVLM-256M (`idefics3`) | `config.json` → **`image_token_id`** = 49190 (`<image>`); `<fake_token_around_image>` = 49189, `<global-img>` = 49152 | **`generation_config.json` → `eos_token_id` = 49279** (`<end_of_utterance>`, which is also `tokenizer_config.eos_token`). `config.text_config.eos_token_id` = 2 is `<|im_end|>` and is **wrong** for chat. | `tokenizer_config.bos_token` = `<|im_start|>` (1). The template emits it literally; `generation_config.bos_token_id` = 0 is inconsistent, so ignore it. | `tokenizer_config.json` → `chat_template`. The content must be a list of `{type:text|image}` | `processor_config.json` → `image_seq_len` = 64 per tile, and `preprocessor_config` `do_image_splitting` with `<row_i_col_j>` tokens |
| gemma-3-4b-it (`gemma3`) | `config.json` → `image_token_index` = 262144, `boi_token_index` = 255999, `eoi_token_index` = 256000; `tokenizer_config.image_token` = `<image_soft_token>` | `config.json` / `generation_config.json` → `eos_token_id` = [1, 106] | `tokenizer_config.bos_token` = `<bos>` (2) | `chat_template.jinja` | `config.mm_tokens_per_image` = 256 |

Recommended lookup order:
- image id: `config.image_token_index`, then `config.image_token_id`, then tokenize `processor_config.image_token`.
- stops: `generation_config.eos_token_id`, then `config.eos_token_id`, then `config.text_config.eos_token_id`. Reuse `readStopTokens` from QwenLLM.cpp:24 and accept lists.
- template: via `OrtxApplyChatTemplate`, exactly as `QwenLLMInference::applyChatTemplate` does, with ChatML as the fallback.
- Drop the string-stripping at 298-310, since only generated tokens are decoded anyway.

Root cause: the helper was written for one model family. The ids, template, preprocessing (no resize, mean 0 / std 1), single-sentinel feature splice and positional decoder binding are all FastVLM/LLaVA-Qwen2 assumptions.

Fix:
- **Minimal, and all F1 strictly asks for.** In the ctor, read `image_token_index|image_token_id`, the stop list and the chat template as above, and replace the constants with members. Make EOS a `std::vector` and check both 151645 and 151643 for FastVLM. The effect today is none for FastVLM; it only helps future FastVLM-like LLaVA exports, such as other FastVLM sizes, which are also Qwen2 with the same ids.
- **Out of scope for F1, needed for SmolVLM and gemma.** This belongs in a separate ledger entry:
  - per-architecture preprocessing: resize to `preprocessor_config.size`, the `image_mean`/`image_std` values, SmolVLM tiling plus `pixel_attention_mask`;
  - prompt expansion (`<fake_token_around_image><global-img><image>×64…`, `<start_of_image><image_soft_token>×256<end_of_image>`);
  - by-name decoder binding as in L1.
- Side finding: the FastVLM preprocessing does not resize to 1024 as the HF `CLIPImageProcessor` (shortest_edge 1024, crop 1024) does. It feeds the native size, so a 256² input gives 16 image tokens instead of 256. It works, but it is off-spec.

Risk / blast radius: `Onnx::FastVLMInference` only, used by the "Vision Language Model" node and `fastvlm_smoke`.
Test: `fastvlm_smoke FastVLM-0.5B-ONNX` must be unchanged. Add a check that the ids read from the config equal the old constants.
Effort: **S** for reading ids and template; **L** for real multi-VLM support.

---

### F2 — the q4f16 (and fp16) FastVLM vision encoder logs a "Tensor type mismatch" before the BASIC retry succeeds
Status: **CONFIRMED**.
Evidence:
- `fastvlm_smoke FastVLM-0.5B-ONNX q4f16` shows:
  1. ORT's own `[E:onnxruntime:, inference_session.cc:2811] Exception during initialization: … tensor.h:210 … Tensor type mismatch. T !=…MLFloat16`, in red, from the ORT logger;
  2. our line `Onnxruntime: session init failed (…); retrying with basic graph optimizations`;
  3. `OK … response: The image displays a gradient…`.
- In Python ORT at EXTENDED, only `vision_encoder_fp16` and `vision_encoder_q4f16` fail. `embed_tokens_*` and `decoder_model_merged_*` load fine.
- The fallback is in `Onnx/helpers/OnnxContext.hpp` at `create_session_with_fallback`, just after the comment block that cites this exact model. It catches *any* `Ort::Exception` and retries with `create_session_options(Options{})` at BASIC.

Root cause: an ORT extended-level fusion kernel is float-only and touches an fp16 initializer in the MobileCLIP vision encoder. It is an upstream ORT bug. The noise comes from ORT's default logger at the ERROR level, plus our own fprintf.

Fix, in `create_session_with_fallback`:
- Make the first attempt on `sessionOptions.Clone()` with `SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL)`, so ORT's red E line is suppressed.
- Keep one short line of our own on retry, or print it only if the retry also fails.
- Choosing BASIC upfront for the vision encoder is not clearly better. Measured on CPU, 1024², 8 threads: fp32 extended 2282 ms against basic 2720 ms; q4 extended 3305 ms against basic 2788 ms. It would also need a dtype probe before the session exists, so muting the first attempt is simpler.
- Related issue, worth a ledger line: the retry uses `create_session_options(Options{})`, not the caller's options, so a chosen provider or device is lost on fallback. It also retries on unrelated failures such as a missing file, which duplicates error output.

Risk / blast radius: every caller of `create_session_with_fallback`, which is only QwenLLM.cpp:88 and FastVLM.cpp:151/153/155. Muting only the first attempt does not hide real failures, because the retry logs normally.
Test: `fastvlm_smoke FastVLM-0.5B-ONNX q4f16 fp16` should show no `[E:onnxruntime` line on stderr and still answer.
Effort: **S**.


## Diagnosis — group `pose` (P1–P11, PD1–PD11)

Tree: `~/ossia/score-workshop/src/addons/score-addon-onnx` (read-only). Score was not launched.
Repro scripts are in `diag/pose/`:
- `rep.py`: a shared replica of `letterboxToTensor` (scale = min, round, top-left or centered, pad 0, then (px-mean)/std).
- `p1.py`, `p2.py`, `p3.py`/`p3b.py`, `p4.py`/`p4c.py`, `p7.py`/`p7b.py`/`p7c.py`, `pd10.py`: the per-bug replicas.
- `cls.cpp`: the **real** `Onnx::classify` from `ModelRole.hpp`, compiled standalone with `clang++ -std=c++20` and fed exact model I/O shapes.

Test images:
- `ossia-detection-model-pack/test_images/{body.jpg,hand.jpg,face.png}` (libreonnx).
- ailia `hand_recognition/blazehand/person_hand.jpg`.
- `demo1.png` was not useful: it shows upside-down roller-coaster riders.

---

### P1 — Gold-YOLO head (PINTO 423) fed raw 0-255, so it outputs garbage
Status: CONFIRMED

Evidence:
- `PoseDetector_detect.cpp:148-151`: the MultiClassDetector path always uses `normMeanStd(NchwBgr, {0,0,0}, {1,1,1})`.
- `ModelRole.hpp:304` claims "Covers Gold-YOLO".
- `p1.py` (C++ replica: top-left letterbox, BGR, decodeMultiClass, thr 0.4) on body.jpg:
  ```
  gold_yolo_n_head_post_..._480x640  raw : 20 rows, 20 >= 0.4, all score 1.0, degenerate boxes e.g. [130,17,125,-38] (x2<x1)
  gold_yolo_n_head_post_..._480x640  /255: 7 rows, 4 >= 0.4: heads 0.93/0.90/0.84/0.46 at sane positions
  gold_yolo_m_head_post_..._384x640  raw : 50 rows, all 1.0 garbage;  /255: 4 sane heads 0.93/0.93/0.90/0.64
  426 yolox_n_body_head_hand_post, det-bhh-yolox-n-480x640, det-body-yolox-bhh-320:
       raw: 11-20 sane bodies (0.83-0.90);   /255: 0 rows
  ```

Root cause:
- The PINTO YOLOX body/head/hand "post" exports expect raw 0-255 BGR. Gold-YOLO post exports expect [0,1].
- The branch hard-codes raw. Raw input saturates Gold-YOLO: every row scores 1.0 with inverted boxes.

Fix:
- In `runDetector`'s MultiClassDetector branch (`PoseDetector_detect.cpp` ~l.148), choose the std per export.
- Cheapest discriminator on the available files: output column order.
  - Gold-YOLO outputs `batchno_classid_x1y1x2y2_score` (`sbb == false`).
  - Every raw-input YOLOX/YOLOv9 export here outputs `..._score_x1y1x2y2`.
- Sketch: compute `sbb` before inference, then `std = sbb ? {1,1,1} : {255,255,255}`.
- A more robust option adds a one-shot sanity probe: if at least 90% of rows score ≥ 0.99 or have x2 < x1, switch to /255 and latch the choice.
- Optionally expose a "Detector Normalization" (Auto / 0-255 / 0-1) combobox. It would also serve P3.

Risk / blast radius:
- Every MultiClassDetector preset uses this path: box-bhh*, lm-*-yolox-bhh, det-bhhf/wholebody12.
- None of them has the `_score` suffix, so they are unaffected.

Test: extend `tests/validate_decoders.py` (or `p1.py`) with the gold_yolo files. Assert fewer than 10 rows, scores < 0.99, and x2 > x1.

Effort: S

### P2 — Dynamic-input MultiClassDetector (YOLOv9-Wholebody25, PINTO 459) falls back to 128×128
Status: PARTIAL. The fallback is real, but the ledger's symptom and its suggested fix are both wrong.

Evidence:
- `PoseDetector_detect.cpp:70`: `const int model = role.input_w > 0 ? role.input_w : 128;` (input is `[1,3,'H','W']`, so input_w is -1).
- l.143-144: `mh = role.input_h > 0 ? … : model` gives 128×128.
- l.163-164: decode divides boxes by `mw, mh`.
- The real classify (`cls.cpp`) gives `MultiClassDetector in=-1x-1`.
- Graph inspection:
  - The first node is `prep/Resize(input_bgr → sizes=[1,3,128,160])`.
  - It is followed by a BGR→RGB slice and `Mul 0.003921569`.
  - So the model resizes internally to 160×128, normalizes itself, and emits boxes **in 160×128 pixel space whatever the input size is**.
- `p2.py`, body.jpg (950×641), class 0 ≥ 0.4:
  ```
  128x128 raw: 3 bodies, but x stretched ×1.25 (e.g. x2=1151 > image width 950)
  160x128 raw: 3 bodies, correct boxes [492,16,657,496] [684,15,938,612] [111,87,437,629]
  320x320 raw: 3 bodies, boxes shrunk ×0.5 / ×0.4
  640x480/640x640 raw: shrunk ×0.25
  any size /255: 0 rows
  ```

Root cause:
- The detector does find bodies at 128×128; "finds nothing" did not reproduce on body.jpg.
- The actual bug: this export's output coordinates are in its baked internal 160×128 space. The node normalizes them by the tensor size it fed, so every size except 160×128 misplaces the boxes. At the 128 fallback the x-axis is off by 1.25×.
- The ledger's suggestion ("fall back to 320 or 640") would make it worse: boxes would shrink 2–4×.

Fix, in `runDetector` for MultiClassDetector with a dynamic input:
- Default to 160×128 (w×h). That is the native size of every PINTO 426/434/449/459 post export.
- Better: add an optional "Detector Resolution" override.
- Or parse `_1x3x<H>x<W>` from the model filename when the dims are dynamic, if the filename is reachable from `m_last_det_model`.
- Keep raw 0-255 input: the graph does its own /255.

Risk / blast radius: only dynamic-input MultiClassDetector models. All shipped presets use fixed-size exports.

Test: `p2.py` → at the chosen fallback, the boxes must lie inside the image and match the 160×128 reference within 5 px.

Effort: S

### P3 — YoloxDetector always /255, so ailia `yolox_*.opt` (raw 0-255) gives 0 detections
Status: CONFIRMED

Evidence:
- `PoseDetector_detect.cpp:192-195`: `normMeanStd(NchwBgr, {0,0,0}, {255,255,255})`, with the comment "This PINTO YOLOX-COCO export wants BGR [0,1]".
- `p3.py`/`p3b.py` (grid decode as in `decodeYoloxGrid`, top-left letterbox). Counts are anchors with obj·cls > 0.3; max is the maximum score.

| model | raw 0-255 | /255 |
|---|---|---|
| ailia yolox_s.opt 640, body | 70 (max 0.94) | 0 (max 0.000) |
| ailia yolox_tiny.opt 416, body/face/hand | 63/9/14 | 0/0/0 (max ≤ 0.001) |
| PINTO det-coco-yolox-416, body | 69 (0 persons, garbage) | 35 (persons 0.86) |
| det-coco-yolox-nano-480x640, body | 106 (persons ≤ 0.38) | 56 (persons 0.87) |

Root cause: the two families differ in input range, and the branch hard-codes /255.

Fix:
- Add a latched auto-probe in the YoloxDetector branch.
  - Run /255 first. If `max(obj*cls) < 0.01` over the whole grid, re-run raw on that frame.
  - If raw gives a max above 0.3, latch `m_yolox_raw = true` until the model changes.
- The signal is one-sided and clean: ailia models give exactly ~0 under /255 on every image tried, while the PINTO ones stay ≥ 0.5.
- Or reuse the "Detector Normalization" combobox from P1.
- Model names do not discriminate: `det-coco-yolox-nano-480x640` also has `images`/`output`.

Risk / blast radius: box-coco-*, lm-*-animal-coco and lm-rtmpose-coco (det-coco-yolox-416). The probe keeps /255 for all of them.

Test: `p3b.py` → assert that the chosen mode yields ≥ 1 person ≥ 0.5 on body.jpg for all four models.

Effort: S

### P4 — blazepalm 256 (wild2) has 2944 anchors, but `palmParams(256)` generates 3584
Status: CONFIRMED

Evidence:
- `Detection.hpp:252-255`: `palmParams` always uses strides `{8,16,16,16}`, which at 256 gives 32²·2 + 16²·6 = 3584.
- `PoseDetector_detect.cpp:325-327`: the palm candidate list is `{palmParams(model)}` only, so the count check (l.333-338) falls back to `.front()`.
- The model outputs `regressors [1,2944,18]`.
- ailia's own `anchors.npy` (2944×4, all w=h=1) equals `generateAnchors({256,{8,16,32,32,32}})` exactly (max diff 0.0).
- `p4c.py` on person_hand.jpg (full frame): the best anchor idx 2380 (score 1.0) decodes to palm center (123.6, 56.9) with the current anchors, versus (107.6, 168.9) with the correct ones. That is 112 px off in the 256 input.
- Detections from the stride-8 layer (idx < 2048) are identical, which is why small palms (hand.jpg, body.jpg) looked fine.

Root cause: only one anchor config is tried for palms. The 256 "full" palm model uses the 5-layer `{8,16,32,32,32}` layout.

Fix, in `PoseDetector_detect.cpp` `case PalmDetector:`:
```cpp
candidates = {Onnx::Detection::palmParams(model),
              Onnx::Detection::SsdParams{.input_size = model, .strides = {8,16,32,32,32}}};
```
The count match then picks it. 128 → 896 and 192 → 2016 still select the first candidate.

Risk / blast radius:
- Palm detectors only.
- The anchors are cached per (kind, input, num_boxes), so there is no per-frame cost.

Test: `p4c.py` → assert that the idx-2380 detection maps near (108,169) model px. Alternatively, a unit check `anchorCount == 2944` for the palm-256 candidates.

Effort: S

### P5 — SimCC with dynamic K (RTMW3D, DWPose) classified K=17, so Auto draws with RTMPose_COCO
Status: CONFIRMED

Evidence:
- `ModelRole.hpp:385-391`: `nk = 17`, then it takes the first `shape[1] > 0`. For these models `shape[1]` is dynamic (`MatMulsimcc_x_dim_1`), so nk stays 17.
- `cls.cpp`: rtmw3d_l_384x288 and dwpose-l-384x288 both give `SimccPose Body K=17`.
- The real run gives `(1,133,576)`, `(1,133,768)[, (1,133,576)]`.
- `PoseDetector_internal.hpp:152-153`: `num_keypoints > 50 ? RTMPose_Whole : RTMPose_COCO`, so the result is RTMPose_COCO.
- `PoseDetector_draw.cpp:445-456`: in RTMPose_COCO the skeleton is drawn only for num_kps ∈ {21, 68, 17}. At 133 there are dots only, no skeleton.
- The decode itself is fine: `RTMPose.hpp:184/200` reads K from the runtime shape.

Root cause: the draw workflow is fixed from the declared K at load time, and the declared K is dynamic here.

Fix, pick one:
- (a) In `PoseDetector.cpp` Auto mode, after the first landmark inference, update `m_landmark_role.num_keypoints` from the decoded count and re-derive `draw` (cache it per model).
- (b) Make the draw code choose by `num_kps` when the workflow is RTMPose_COCO and `num_kps == 133` (treat it as Whole).

(b) is smallest: in `draw.cpp` RTMPose_COCO, add `else if(num_kps == 133) drawConnections(Skeletons::wholebody_body, 17);` and use the whole-body palette and radii. (a) is more correct, because it also fixes colors and radii.

Risk / blast radius: Auto-workflow SimCC presets. PD2 is the data-side workaround.

Test: add rtmw3d/dwpose shapes to the classify test, and a draw test with 133 keypoints under Auto.

Effort: S (b) / M (a)

### P6 — Multi-class body/head/hand detectors can't serve as the *hand* detector in two-stage mode
Status: CONFIRMED (static; the class ids were confirmed at runtime)

Evidence:
- `PoseDetector.cpp:338-339` (single) and `PoseDetector_tracking.cpp:347` (multi) call `runDetector(..., role.domain, -2, …)`, i.e. keep_class = -2.
- `PoseDetector_detect.cpp:162`: `const int keep = (keep_class == -2) ? 0 : keep_class;`. The MultiClassDetector branch ignores `target`.
- The YoloxDetector branch (l.185-188) maps only Animal and ignores Hand.
- `p1.py` confirms that the bhh exports label 0 = body, 1 = head, 2 = hand (e.g. det-bhh on demo1: classes 1 and 2 are small head/hand boxes).
- `inputs.detection_class` is only read by `runBoxDetection` (`detect.cpp:413`).

Root cause: the two-stage paths never forward a class choice, and the domain→class map exists only for YOLOX Animal.

Fix:
- In both two-stage call sites, pass `detection_class >= 0 ? detection_class : -2`.
- In the MultiClassDetector branch, when keep_class == -2, map by `target`: Hand → 2, Face → 3 for the bhhf 4-class export. Keep Body → 0.
- Wholebody12/25 use different ids, so the explicit Detection Class override is what matters there.
- Update the Detection Class port description, which currently says "Box Detection" only.

Risk / blast radius:
- Every two-stage preset uses keep -2 with a default of 0.
- Honouring the port only when ≥ 0 keeps all current presets unchanged (they store -1).

Test: a two-stage hand pipeline (lm-hand-rtmpose-21 + det-body-yolox-bhh-320, Detection Class = 2) on hand.jpg or body.jpg should yield a hand crop.

Effort: S

### P7 — `mediapipeRect` assumes BlazeFace keypoint order, so RetinaFace crops are misaligned
Status: REFUTED (the order hypothesis). A minor size difference remains.

Evidence:
- `ROI.hpp:45-50` / `55-61`: face/mobileFace use kp 0 → 1 with target angle 0; center and size come from the box.
- `p7.py` compares BlazeFace-128 and RetinaFace-mobile (C++ replicas) on face.png rotated by 0/30/-45°:
  - Both put the image-left eye at index 0 and the image-right eye at index 1.
  - The ROI angles agree within 3° (−5.2/−2.3, −32.4/−32.1, 42.0/42.0).
  - The centers agree within about 20 px (image 640 wide).
  - The RetinaFace box is taller (518×634 vs 561×561), so its ROI is 13–22% larger.
- `p7b.py`: FaceMesh-468 run on both crops. Landmarks differ by 8.0/5.1/10.4 px mean at an interocular distance of about 340 px (≤ 3% IOD). The face flag is 1.0 on both.
- `p7c.py`: MobileFaceNet (scale 1.1, tight crop) differs by 28/17/25 px, about 5–7% of the outer-eye distance. This effect comes from the box size, not from the keypoint order.
- At 180° RetinaFace still reports image-left-eye-first, so the ROI is upright on an upside-down face. That is a property of RetinaFace's landmarks, not of the ROI code.

Root cause: no ordering bug. RetinaFace's 5-point order (L-eye, R-eye, …) already matches BlazeFace kp0/kp1 in image terms.

Fix:
- None is needed for FaceMesh.
- Optional for MobileFaceNet: a per-detector size factor, e.g. when the detector role is RetinaFace/FaceBoxes, use `size = max(w,h)*0.9` in `detectionRect`. This needs access to the detector role there.

Risk / blast radius: n/a

Test: `p7b.py` / `p7c.py` are the regression checks.

Effort: S (optional tweak)

### P8 — HRNet Animal-Pose (20 kpts) drawn with the AP-10K 17-point skeleton
Status: CONFIRMED (static)

Evidence:
- `pose_estimation__animalpose__hrnet_w32_256x256.onnx` has `heatmap [1,20,64,64]` and is square. `cls.cpp` gives `HeatmapPose Animal K=20`, so the workflow is AnimalPose (`internal.hpp`, HeatmapPose case).
- `draw.cpp:516-517`: `drawConnections(Skeletons::ap10k, 17)`. The AP-10K edges (l.117-124: 0 L-eye, 2 nose, 3 neck, 4 tail-root, …) are applied to the Animal-Pose order:
  - Animal-Pose order: 0 L-eye, 1 R-eye, 2 L-ear, 3 R-ear, 4 nose, 5 throat, 6 tail, 7 withers, 8-11 elbows, 12-15 knees, 16-19 paws.
  - Resulting wrong edges: nose→R-B-elbow (4→11), R-ear→throat (3→5), and so on.
  - Paws 17-19 are never connected.

Related bonus finding: ViTPose `ap10k`/`apt36k` exports are 256×192 (not square). They classify as `HeatmapPose Body K=17` (`cls.cpp`), so Auto draws them with the **human COCO-17** skeleton. The shipped presets avoid this only because they force Workflow 6.

Root cause: skeleton selection is by workflow only, never by K, dataset, or the animal-ness of non-square inputs.

Fix:
- Add `Skeletons::animalpose20` (mmpose animalpose skeleton, 0-based):
  `{0,1},{0,2},{1,3},{0,4},{1,4},{4,5},{5,7},{6,7},{5,8},{8,12},{12,16},{5,9},{9,13},{13,17},{6,10},{10,14},{14,18},{6,11},{11,15},{15,19}`.
- In the AnimalPose case, pick `num_kps == 20 ? animalpose20 : ap10k`.
- Add a `getColor` variant for it.

Risk / blast radius: AnimalPose drawing only.

Test: a draw unit test with K=20, or a visual check with `p*`-style overlay.

Effort: S

### P9 — ViTPose aic (14) / mpii (16) / coco_25 (25) get dots only
Status: CONFIRMED (static)

Evidence:
- `cls.cpp` gives `HeatmapPose Body K=14/16/25`, so the workflow is ViTPose.
- `draw.cpp` has no `case ViTPose` in the connection switch, so it falls to `default:` (l.519-527), which draws only when num_kps == 17 or 68.

Root cause: no topology tables for AIC-14, MPII-16 or COCO-25 (Halpe-style body + feet).

Fix: add three edge tables and select them by num_kps in the default/ViTPose branch:
- AIC-14 order: 0 R-sho, 1 R-elb, 2 R-wri, 3 L-sho, 4 L-elb, 5 L-wri, 6 R-hip, 7 R-knee, 8 R-ank, 9 L-hip, 10 L-knee, 11 L-ank, 12 head-top, 13 neck.
  Edges: {12,13},{13,0},{0,1},{1,2},{13,3},{3,4},{4,5},{0,6},{3,9},{6,7},{7,8},{9,10},{10,11},{6,9}.
- MPII-16 order: 0 R-ank … 5 L-ank, 6 pelvis, 7 thorax, 8 neck, 9 head-top, 10 R-wri … 15 L-wri.
  Edges: {0,1},{1,2},{2,6},{3,6},{3,4},{4,5},{6,7},{7,8},{8,9},{7,12},{12,11},{11,10},{7,13},{13,14},{14,15}.
- COCO-25 = COCO-17 + {17 neck?}. Verify the order against ViTPose's `coco_25` config before adding. It was **not verified** here.

Risk / blast radius: drawing only.

Test: a draw test per K.

Effort: S–M

### P10 — 3DDFA FaceBoxesProd classifies only thanks to ORT shape inference
Status: CONFIRMED

Evidence:
- `onnx.load` shows the declared outputs `output [dyn,dyn,dyn]`, `367 [dyn,dyn,dyn]`, and input `[dyn,3,dyn,dyn]`.
  - ORT with ENABLE_ALL reports `[…,4]` / `[…,2]`.
  - ORT with `ORT_DISABLE_ALL` reports all three dims dynamic. That is the setting `OnnxContext.hpp:180` uses for OpenVINO.
- Real classify (`cls.cpp`):
  ```
  FaceBoxesProd ORT_ENABLE_ALL  -> FaceBoxesDetector
  FaceBoxesProd ORT_DISABLE_ALL -> Unknown
  ```
- Mechanism in `ModelRole.hpp:160-162`: `last = shape.back()>0 ? … : 1`. That makes both outputs look like `[.., 1]` score tensors, so the A0 rule (l.264-277) sees last4 = last2 = 0.
- The same file ships as `pose-detector/detectors/det-face-faceboxes.onnx` (md5 identical).
- The input is always dynamic, so it runs at the 320 fallback (`detect.cpp:227-228`) in every case. That part is by design.

Root cause: the classifier depends on output dims the file does not declare.

Fix, in `PoseDetector.cpp` when (re)loading a model:
- If `classify` returns Unknown and the model has a rank-4 image input with any dynamic output dim, run one dummy inference at 320×320.
- Rebuild `ModelIO.outputs[i].shape` from the actual output tensors and classify again.
- This covers every "all-dynamic" export, not just FaceBoxes.
- Alternative: a name fallback, but "output"/"367" are useless names.

Risk / blast radius: load time only (one extra inference, only for Unknown models).

Test: a `cls.cpp`-style unit test with all-dynamic shapes plus the probe path, or run the probe in Python with `ORT_DISABLE_ALL`.

Effort: M

### P11 — `YOLO_pose` only decodes 56×8400 and 300×57, K fixed to 17
Status: CONFIRMED (static; no non-640 or non-17 YOLO-pose model was available to run)

Evidence:
- `Yolo.hpp:281`: `NUM_KPS = 17`.
- l.349: `if (Nfloats == 56 * 8400)`.
- l.407: `else if (Nfloats == 300 * 57)`. Everything else silently yields nothing.
- `PoseDetector_landmark.cpp:741-747`: the multi-instance path hard-codes 17 keypoints.
- Classify accepts much more (`ModelRole.hpp:441-456`: any `(d-5)%3==0` or `(d-6)%3==0`, 5 ≤ K ≤ 60).
- `cls.cpp`: a yolov8n-pose exported at 320 (`[1,56,2100]`) gives `YoloPose K=17`, then the decode gives 0 poses.
- Extra: a YOLOv8 **hand**-pose (K=21, `[1,68,8400]`) classifies as **MobileFaceNet K=68**. The `shape[1]==68` rule at `ModelRole.hpp:428-432` runs before the YOLO rule F.
- A transposed `[1,8400,56]` export would pass the Nfloats check but be read as channel-major (wrong).

Root cause: the decoder is keyed on element count for two specific exports, not on the output shape.

Fix: rewrite `YOLO_pose::processOutput` to take the shape.
- `[1,F,A]` with F < A means channel-major v8/v11: `K = (F-5)/3` and anchors = A.
- `[1,N,F]` with N ≤ 300 and `(F-6)%3 == 0` means yolo26 row-major: `K = (F-6)/3`.
- Store keypoints in a `std::vector` sized K instead of `[17][3]`.
- In `runYOLOPose` use `K` instead of 17.
- In `classify`, run rule F (YOLO-pose) before the MobileFaceNet `shape[1]==68` rule, or require that rule to have rank-2/3 small outputs.

Risk / blast radius: every YOLO-pose preset (ss-yolov8*, ss-yolo26*, reid-yolopose). The 640/17 behaviour must stay bit-identical.

Test:
- Add synthetic-tensor unit tests for `[1,56,2100]`, `[1,68,8400]` and `[1,300,57]`.
- Plus the real yolov8n/yolo26n on body.jpg against the current output.

Effort: M

---

## Preset data (PD1–PD11)

Port ids in `.scp` `Preset`: 1 Landmark Model, 2 Workflow, 4 Min Confidence, 7 Detection Model, 14 Re-ID Model, 17 Re-ID Preprocess, 19 Detection Class, 30 Detection Hold.

Workflow enum values:

| Int | Workflow |
|---|---|
| 0 | Auto |
| 1 | BlazePose |
| 2 | RTMPose_COCO |
| 3 | RTMPose_Whole |
| 4 | ViTPose |
| 5 | YOLOPose |
| 6 | AnimalPose |
| 7 | MediaPipeHands |
| 8 | FaceMesh |
| 9 | BlazeFace |
| 10 | MobileFaceNet |
| 11 | RTMPoseFace |
| 12 | BoxDetection |

Directory: `~/Documents/ossia/score/packages/pose-detector/presets/PoseDetector/` (58 files, all with 29 ids 1..29).

- **PD1: CONFIRMED.** No preset has id 30.
  - Edit: append `, [30, {"Int": 6}]` before the closing `]]` of `"Preset"` in all 58 files.
- **PD2: CONFIRMED.** Both have `[2, {"Int": 0}]`, which P5 turns into RTMPose_COCO for K=133.
  - Edit in `lm-rtmw3d-yolox-3d.scp` and `wholeframe-rtmw3d-wholeframe.scp`: `[2, {"Int": 0}]` → `[2, {"Int": 3}]`.
- **PD3: CONFIRMED.** `lm-facemesh.scp` has `[4, {"Float": 1.0}]`.
  - `detThreshold()` clamps it to 0.95 for the BlazeFace detector. BlazeFace scores 0.85 on face.png, so there is no detection.
  - Separately, FaceMesh `face_flag < 1.0` fails the presence gate.
  - Edit: `[4, {"Float": 1.0}]` → `[4, {"Float": 0.5}]`.
- **PD4: CONFIRMED.** `reid-facemesh-facereid.scp` points 14 at `reid-body-market1501-resnet50.onnx`.
  - `reid-face-retail-0095.onnx` (`[1,3,128,128] → [1,256]`) exists in `/mnt/win2/models/models-presets/models/pose-detector/reid/`. Copy it into `packages/pose-detector/reid/`.
  - Edit: `[14, {"String": "<LIBRARY>:packages/pose-detector/reid/reid-face-retail-0095.onnx"}]`, and `[17, {"String": "Auto"}]` → `[17, {"String": "RawBGR"}]`.
  - Auto would already resolve 128×128 to RawBGR; the explicit value documents it.
- **PD5: CONFIRMED.** `reid-{blazepose,rtmpose,vitpose,yolopose}-osnet.scp` all use `reid-body-market1501-resnet50.onnx` (89.6 MB).
  - OSNet is available: `reid-body-osnet-x0_5-msmt17.onnx` is 2.4 MB, `x1_0` is 8.3 MB (the ledger said 8.7). Both are `[N,3,256,128] → [N,512]`; Auto picks ImageNetRGB, which is correct.
  - Edit (after copying into the package): `[14, {"String": "<LIBRARY>:packages/pose-detector/reid/reid-body-osnet-x0_5-msmt17.onnx"}]`.
  - Or rename `"Name"` to "ReID … ResNet50".
- **PD6: CONFIRMED.** `lm-rtmpose-coco.scp` uses `lm-body-rtmpose-wb133.onnx` (`[1,133,…]`).
  - What distinguishes it from `lm-rtmpose-yolox` is the COCO-grid detector `det-coco-yolox-416`.
  - Edit: `"Name": "RTMPose COCO"` → `"Name": "RTMPose WholeBody (COCO detector)"`. Optionally rename the file to `lm-rtmpose-wholebody-cocodet.scp`.
- **PD7: CONFIRMED.** `box-face-faceboxes.scp`, `box-face-faceboxes-pinto.scp` and `box-face-retinaface.scp` have `[2, {"Int": 0}]`. They work today only via the `Auto && !have_landmark` box-only fallback (`PoseDetector.cpp:188-192`).
  - Edit: `[2, {"Int": 0}]` → `[2, {"Int": 12}]` in the three files.
- **PD8: CONFIRMED.** Box presets are spread over "Detection (boxes)", "Face/Detection" and "Hand". `det-blazeface` and `det-palm` (detector-as-pose) sit in "Face/Landmarks" and "Hand".
  - Suggested edits:
    - `box-hand-palm`, `box-hand-rtmdet`: `"Category": "Hand"` → `"Category": "Detection (boxes)"`.
    - `box-face-*` (4 files): `"Face/Detection"` → `"Detection (boxes)"`.
  - The Body/3D, Body/Whole-frame and Body/Single-stage categories are already consistent.
  - An alternative: keep a domain split and add `Detection (boxes)/Body|Face|Hand`, whichever scheme the browser nests best.
- **PD9: CONFIRMED.**
  - `BlazePose Good.scp` has conf 0.1252223700284958 and **no detector** (7 empty). It duplicates `lm-blazepose-full` without its detector.
  - `YOLOPose.scp` has conf 0.08445216715335846. It duplicates `ss-yolov8n`.
  - Recommend deleting both. If they are kept: `[4, {"Float": 0.3}]` and new names.
- **PD10: CONFIRMED.**
  - The conf values in question:
    - `lm-blazepose-full`: 0.6879.
    - `lm-rtmpose-wholebody-full` and `lm-vitpose-wholebody-full`: 0.6879.
    - `lm-hand-rtmpose-full`: 0.5546.
    - `lm-blazehand`: 0.6046.
  - They feed `detThreshold()` (clamped [0.05, 0.95]; the node uses 0.66× that while tracking).
  - `pd10.py` replica scores:
    - BlazePose-128 on body.jpg tops at 0.708: only one anchor passes 0.688, so the margin is thin.
    - RTMDet-hand-320 tops at 0.26 (body) / 0.43 (hand): **nothing** passes 0.5546.
    - Palm-128: 0.75–0.84, fine.
    - YOLOX-416 persons: 0.75–0.93.
  - Edit:
    - `[4, {"Float": 0.6879374980926514}]` → `[4, {"Float": 0.5}]` in lm-blazepose-full, lm-rtmpose-wholebody-full and lm-vitpose-wholebody-full.
    - `[4, {"Float": 0.554604172706604}]` → `[4, {"Float": 0.4}]` in lm-hand-rtmpose-full. The RTMDet scores seen here suggest even 0.3 may be needed.
    - `[4, {"Float": 0.604604184627533}]` → `[4, {"Float": 0.5}]` in lm-blazehand.
- **PD11: CONFIRMED.** `~/Documents/ossia/score/packages/user/presets/PoseDetector/` has 54 files, with old port layouts of 17 or 20 ids (vs 29/30 now).
  - They contain 79 model paths that do not exist (`/home/jcelerier/projets/celtera/libreonn…`, `/mnt/win1/PINTO_model_zoo/…`).
  - Action: delete the directory (`rm -r …/user/presets/PoseDetector`). These are user-directory files, so confirm with the user first.
