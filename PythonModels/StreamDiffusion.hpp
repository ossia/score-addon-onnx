#pragma once
#include <cmath>
#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <halp/sample_accurate_controls.hpp>
#include <halp/texture.hpp>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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
      void update(StreamDiffusionImg2Img& self) { self.m_needsPrepare = true; }
    } prompt;
    struct : halp::lineedit<"Prompt -", "anime">
    {
      void update(StreamDiffusionImg2Img& self) { self.m_needsPrepare = true; }
    } negative_prompt;
    struct : halp::lineedit<"Model", "SimianLuo/LCM_Dreamshaper_v7">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_needsModel = true;
        self.m_needsCreate = true;
        self.m_needsPrepare = true;
      }
    } model;
    struct : halp::lineedit<"LCM", "latent-consistency/lcm-lora-sdv1-5">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_needsCreate = true;
        self.m_needsPrepare = true;
      }
    } lcm;
    struct : halp::lineedit<"VAE", "madebyollin/taesd">
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_needsCreate = true;
        self.m_needsPrepare = true;
      }
    } vae;
    struct : halp::spinbox_i32<"Seed", halp::free_range_max<>>
    {
      void update(StreamDiffusionImg2Img& self) { self.m_needsPrepare = true; }
    } seed;
    struct : halp::spinbox_i32<"Steps", halp::range{1, 100, 50}>
    {
      void update(StreamDiffusionImg2Img& self) { self.m_needsPrepare = true; }
    } steps;
    struct : halp::knob_f32<"Guidance", halp::range{0.5, 1.5, 1.0}>
    {
      void update(StreamDiffusionImg2Img& self) { self.m_needsPrepare = true; }
    } guidance;
    struct : halp::spinbox_i32<"T1", halp::range{1, 50, 15}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_needsModel = true;
        self.m_needsCreate = true;
        self.m_needsPrepare = true;
      }
    } t1;
    struct : halp::spinbox_i32<"T2", halp::range{1, 50, 40}>
    {
      void update(StreamDiffusionImg2Img& self)
      {
        self.m_needsModel = true;
        self.m_needsCreate = true;
        self.m_needsPrepare = true;
      }
    } t2;
  } inputs;

  struct
  {
    halp::texture_output<"Out"> image;

  } outputs;

  StreamDiffusionImg2Img() noexcept;
  ~StreamDiffusionImg2Img();

  void operator()();

private:
  void load_model();
  void create();
  void prepare();
  bool m_needsModel{true};
  bool m_needsCreate{true};
  bool m_needsPrepare{true};
  bool m_needsTrain{true};
  bool m_created{false};
  bool m_prepared{false};
};
}
