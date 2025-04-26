#pragma once
namespace PythonModels
{

int python_interpreter_global_instance();

struct StreamDiffusionWrapper
{
  StreamDiffusionWrapper();

  int setup_path();
  int setup_imports();
  void load_model();
  void create();
  void prepare();

  void img2img(
      unsigned char* in_bytes,
      int width,
      int height,
      unsigned char* out_bytes);

  bool prepare_inference();

  inline void set_prompt_positive(std::string v)
  {
    m_prompt_positive = std::move(v);
    m_needsPrepare = true;
  }
  inline void set_prompt_negative(std::string v)
  {
    m_prompt_negative = std::move(v);
    m_needsPrepare = true;
  }
  inline void set_model(std::string v)
  {
    m_model = std::move(v);
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_lcm(std::string v)
  {
    m_lcm = std::move(v);
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_vae(std::string v)
  {
    m_vae = std::move(v);
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }

  inline void set_seed(int64_t v)
  {
    m_seed = v;
    m_needsPrepare = true;
  }
  inline void set_steps(int v)
  {
    m_steps = v;
    m_needsPrepare = true;
  }
  inline void set_guidance(float v)
  {
    m_guidance = v;
    m_needsPrepare = true;
  }
  inline void set_temps(std::vector<int> v)
  {
    m_temps = std::move(v);
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }
  inline void set_size(int w, int h)
  {
    m_width = w;
    m_height = h;
    m_needsModel = true;
    m_needsCreate = true;
    m_needsPrepare = true;
  }

  std::string m_prompt_positive;
  std::string m_prompt_negative;
  std::string m_model;
  std::string m_lcm;
  std::string m_vae;
  std::vector<std::string> m_loras;
  std::vector<int> m_temps;
  int64_t m_seed{};
  int m_steps{};
  int m_width{};
  int m_height{};
  float m_guidance{};

  bool m_needsModel{true};
  bool m_needsCreate{true};
  bool m_needsPrepare{true};
  bool m_needsTrain{true};
  bool m_hasModel{false};
  bool m_created{false};
  bool m_prepared{false};
};

}
