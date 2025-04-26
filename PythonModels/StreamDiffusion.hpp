#pragma once
#include <PythonModels/StreamDiffusionWrapper.hpp>
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>
namespace PythonModels
{
struct StreamDiffusionImg2Img
{
public:
  halp_meta(name, "StreamDiffusion img2img");
  halp_meta(c_name, "streamdiffusion_img2img");
  halp_meta(category, "AI/Generative");
  halp_meta(author, "StreamDiffusion authors, Jean-Michaël Celerier");
  halp_meta(description, "Funky little images.");
  halp_meta(uuid, "a346139a-9e04-4f7b-8ecc-45fd5caea0aa");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/streamdiffusion.html");

  struct
  {
    halp::texture_input<"In"> image;
    struct : halp::lineedit<"Prompt +", "mushroom kingdom, charcoal, velvia">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_prompt_positive(value);
      }
    } prompt;
    struct : halp::lineedit<"Prompt -", "anime">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_prompt_negative(value);
      }
    } negative_prompt;
    struct : halp::lineedit<"Model", "SimianLuo/LCM_Dreamshaper_v7">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_model(value);
      }
    } model;
    struct : halp::lineedit<"LCM", "latent-consistency/lcm-lora-sdv1-5">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_lcm(value);
      }
    } lcm;
    struct : halp::lineedit<"VAE", "madebyollin/taesd">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_vae(value);
      }
    } vae;
    struct : halp::spinbox_i32<"Seed", halp::free_range_max<>>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_seed(value);
      }
    } seed;
    struct : halp::spinbox_i32<"Steps", halp::range{1, 100, 50}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_steps(value);
      }
    } steps;
    struct : halp::knob_f32<"Guidance", halp::range{0.5, 1.5, 1.0}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_guidance(value);
      }
    } guidance;
    struct : halp::spinbox_i32<"T1", halp::range{0, 50, 15}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        if (self.inputs.tcount > 2)
          ts.push_back(self.inputs.t3);
        self.m_wrapper.set_temps(std::move(ts));
      }
    } t1;
    struct : halp::spinbox_i32<"T2", halp::range{1, 50, 40}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        if (self.inputs.tcount > 2)
          ts.push_back(self.inputs.t3);
        self.m_wrapper.set_temps(std::move(ts));
      }
    } t2;
    struct : halp::spinbox_i32<"T3", halp::range{1, 50, 40}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        if (self.inputs.tcount > 2)
          ts.push_back(self.inputs.t3);
        self.m_wrapper.set_temps(std::move(ts));
      }
    } t3;
    struct : halp::spinbox_i32<"T count", halp::range{1, 3, 2}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        std::vector<int> ts;
        ts.push_back(self.inputs.t1);
        if (self.inputs.tcount > 1)
          ts.push_back(self.inputs.t2);
        if (self.inputs.tcount > 2)
          ts.push_back(self.inputs.t3);
        self.m_wrapper.set_temps(std::move(ts));
      }
    } tcount;
    struct : halp::xy_spinboxes_t<int, "Size", halp::range{1, 2048, 512}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_wrapper.set_size(value.x, value.y);
      }
    } size;
    // TODO feed % last output, feed % last input
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

  } outputs;

  StreamDiffusionImg2Img() noexcept;
  ~StreamDiffusionImg2Img();

  void operator()();

private:
  StreamDiffusionWrapper m_wrapper;
};
}
