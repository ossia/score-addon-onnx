#include <QCoreApplication>
#include <QDebug>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv)
{
  qputenv("QT_ASSUME_STDERR_HAS_CONSOLE", "1");
  Ort::InitApi();
  QCoreApplication app(argc, argv);

  if (argc < 2)
  {
    qDebug() << "Usage: inspect_model <model_path>";
    return 1;
  }

  try
  {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_inspector");
    Ort::SessionOptions session_options;
    
    std::string model_path = argv[1];
    qDebug() << "Inspecting model:" << model_path.c_str();
    
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    
    qDebug() << "\n=== INPUT INFO ===";
    qDebug() << "Input count:" << session.GetInputCount();
    
    for (size_t i = 0; i < session.GetInputCount(); i++)
    {
      auto input_name = session.GetInputNameAllocated(i, allocator).get();
      auto input_type_info = session.GetInputTypeInfo(i);
      auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      
      qDebug() << "Input" << i << ":" << input_name;
      qDebug() << "  Shape:";
      for (auto dim : shape)
        qDebug() << "    " << dim;
    }
    
    qDebug() << "\n=== OUTPUT INFO ===";
    qDebug() << "Output count:" << session.GetOutputCount();
    
    for (size_t i = 0; i < session.GetOutputCount(); i++)
    {
      auto output_name = session.GetOutputNameAllocated(i, allocator).get();
      auto output_type_info = session.GetOutputTypeInfo(i);
      auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      
      qDebug() << "Output" << i << ":" << output_name;
      qDebug() << "  Shape:";
      for (auto dim : shape)
        qDebug() << "    " << dim;
    }
    
    return 0;
  }
  catch (const Ort::Exception& e)
  {
    qDebug() << "ONNX Runtime error:" << e.what();
    return 1;
  }
}