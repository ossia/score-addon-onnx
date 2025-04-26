#pragma once
#include <PythonModels/StreamDiffusionWrapper.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>
namespace PythonModels
{
struct StreamDiffusionTxt2Img
{
public:
  halp_meta(name, "StreamDiffusion txt2img");
  halp_meta(c_name, "streamdiffusion_txt2img");
  halp_meta(category, "AI/Generative");
  halp_meta(author, "StreamDiffusion authors, Jean-Michaël Celerier");
  halp_meta(description, "Funky little images.");
  halp_meta(uuid, "ef81de72-ebf9-4ed4-8322-39c29f0ffaa5");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/streamdiffusion.html");

  struct
  {
    struct : halp::lineedit<"Prompt +", "mushroom kingdom, charcoal, velvia">
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_prompt_positive(value);
        self.inputs.trigger.value.emplace();
      }
    } prompt;
    struct : halp::lineedit<"Prompt -", "anime">
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_prompt_negative(value);
        self.inputs.trigger.value.emplace();
      }
    } negative_prompt;
    struct : halp::lineedit<"Model", "SimianLuo/LCM_Dreamshaper_v7">
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_model(value);
        self.inputs.trigger.value.emplace();
      }
    } model;
    struct : halp::lineedit<"LoRAs", "latent-consistency/lcm-lora-sdv1-5">
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_loras(value);
        self.inputs.trigger.value.emplace();
      }
    } loras;
    struct : halp::lineedit<"VAE", "madebyollin/taesd">
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_vae(value);
        self.inputs.trigger.value.emplace();
      }
    } vae;
    struct : halp::spinbox_i32<"Seed", halp::free_range_max<>>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_seed(value);
        self.inputs.trigger.value.emplace();
      }
    } seed;
    struct : halp::spinbox_i32<"Steps", halp::range{1, 50, 50}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_steps(value);
        self.inputs.trigger.value.emplace();
      }
    } steps;
    struct : halp::knob_f32<"Guidance", halp::range{0.5, 1.5, 1.0}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_guidance(value);
        self.inputs.trigger.value.emplace();
      }
    } guidance;
    struct : halp::knob_f32<"Delta", halp::range{-200, 200, 1.0}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_delta(value);
        self.inputs.trigger.value.emplace();
      }
    } delta;
    struct : halp::spinbox_i32<"T1", halp::range{0, 50, 15}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        self.m_wrapper.set_temps(std::move(ts));
        self.inputs.trigger.value.emplace();
      }
    } t1;
    struct : halp::spinbox_i32<"T2", halp::range{1, 50, 40}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        self.m_wrapper.set_temps(std::move(ts));
        self.inputs.trigger.value.emplace();
      }
    } t2;
    struct : halp::spinbox_i32<"T count", halp::range{1, 2, 2}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        self.m_wrapper.set_temps(std::move(ts));
        self.inputs.trigger.value.emplace();
      }
    } tcount;
    struct : halp::xy_spinboxes_t<int, "Size", halp::range{1, 2048, 512}>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_size(value.x, value.y);
        self.inputs.trigger.value.emplace();
      }
    } size;

    struct : halp::toggle<"Add noise", halp::default_on_toggle>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_add_noise(value);
        self.inputs.trigger.value.emplace();
      }
    } add_noise;
    struct : halp::toggle<"Denoising batch", halp::default_on_toggle>
    {
      void update(StreamDiffusionTxt2Img& self)
      {
        self.m_wrapper.set_denoising_batch(value);
        self.inputs.trigger.value.emplace();
      }
    } denoise_batch;
    halp::toggle<"Manual mode"> manual;
    halp::val_port<"Manual trigger", std::optional<halp::impulse>> trigger;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

  } outputs;

  StreamDiffusionTxt2Img() noexcept;
  ~StreamDiffusionTxt2Img();

  void operator()();

private:
  StreamDiffusionWrapper m_wrapper;
};
}
