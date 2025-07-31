#pragma once
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace PythonModels
{

int python_interpreter_global_instance();

struct StreamDiffusionWrapper
{
  StreamDiffusionWrapper();

  int init();
  int setup_path();
  int setup_imports();
  void load_model();
  void create();
  void prepare();
  void update_prompt();

  void img2img(
      unsigned char* in_bytes,
      int width,
      int height,
      unsigned char* out_bytes);
  void txt2img(unsigned char* out_bytes);

  bool prepare_inference();

  inline void set_prompt_positive(std::string v)
  {
    std::lock_guard _{m_mtx};
    m_prompt_positive = std::move(v);
    if (!m_needsPrepare)
      m_needsUpdatePrompt = true;
  }
  inline void set_prompt_negative(std::string v)
  {
    std::lock_guard _{m_mtx};
    m_prompt_negative = std::move(v);
    m_needsPrepare = true;
  }
  inline void set_model(std::string v)
  {
    std::lock_guard _{m_mtx};
    m_model = std::move(v);
    m_sd_turbo = m_model.find("turbo") != std::string::npos;
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }

  inline void set_loras(std::string v)
  {
    std::lock_guard _{m_mtx};

    std::vector<std::string> loras;
    m_loras.clear();
    boost::split(loras, v, boost::is_any_of("\n"));
    for (auto str : loras)
    {
      if (str.empty())
        continue;
      std::vector<std::string> lora;
      boost::split(lora, str, boost::is_any_of("@"));
      if (lora[0].empty())
        continue;
      if (lora.size() == 1)
      {
        m_loras.emplace_back(lora[0], "");
      }
      else if (lora.size() >= 2)
      {
        m_loras.emplace_back(lora[0], lora[1]);
      }
    }
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }

  inline void set_vae(std::string v)
  {
    std::lock_guard _{m_mtx};
    m_vae = std::move(v);
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }

  inline void set_seed(int64_t v)
  {
    std::lock_guard _{m_mtx};
    m_seed = v;
    m_needsPrepare = true;
  }
  inline void set_steps(int v)
  {
    std::lock_guard _{m_mtx};
    m_steps = v;
    m_needsPrepare = true;
  }
  inline void set_guidance(float v)
  {
    std::lock_guard _{m_mtx};
    m_guidance = v;
    m_needsPrepare = true;
  }
  inline void set_delta(float v)
  {
    std::lock_guard _{m_mtx};
    m_delta = v;
    m_needsPrepare = true;
  }
  inline void set_temps(std::vector<int> v)
  {
    std::lock_guard _{m_mtx};
    m_temps = std::move(v);
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_size(int w, int h)
  {
    std::lock_guard _{m_mtx};
    m_width = w;
    m_height = h;
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_add_noise(bool b)
  {
    std::lock_guard _{m_mtx};
    m_add_noise = b;
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_denoising_batch(bool b)
  {
    std::lock_guard _{m_mtx};
    m_denoising_batch = b;
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_cfg(std::string v)
  {
    m_cfg = std::move(v);
    if (m_cfg != "none" && m_cfg != "self" && m_cfg != "full"
        && m_cfg != "initialize")
      m_cfg = "none";
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_lora_weight(float b)
  {
    std::lock_guard _{m_mtx};
    m_lora_weight = b;
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }

  void read_image_pil(std::string_view bytes, unsigned char* output);

  std::string m_prompt_positive;
  std::string m_prompt_negative;
  std::string m_model;
  std::string m_vae;
  std::string m_cfg{"self"};
  std::vector<std::pair<std::string, std::string>> m_loras;
  std::vector<int> m_temps;
  int64_t m_seed{};
  int m_steps{};
  int m_width{};
  int m_height{};
  float m_guidance{};
  float m_delta{};
  float m_lora_weight{};
  bool m_add_noise{true};
  bool m_denoising_batch{true};
  bool m_sd_turbo{false};

  bool m_needsModel{true};
  bool m_needsCreate{true};
  bool m_needsPrepare{true};
  bool m_needsUpdatePrompt{true};
  bool m_needsTrain{true};
  bool m_hasModel{false};
  bool m_created{false};
  bool m_prepared{false};

  std::mutex m_mtx;
};

}
