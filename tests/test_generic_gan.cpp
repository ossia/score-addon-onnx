#include <QCoreApplication>
#include <QDebug>
#include <QFile>

#include "generic_gan.hpp"

#include <iostream>
#include <chrono>

using namespace GANInference;

int main(int argc, char** argv)
{
  qputenv("QT_ASSUME_STDERR_HAS_CONSOLE", "1");
  Ort::InitApi();
  QCoreApplication app(argc, argv);

  try
  {
    qDebug() << "=== Generic GAN Architecture Test ===\n";
#if 1
    // Test FBAnime StyleGAN model
    qDebug() << "Testing FBAnime StyleGAN model...";
    std::vector<std::string> fbanime_paths = {
      "/home/jcelerier/projets/oss/fbanime-gan/g_mapping.onnx",
      "/home/jcelerier/projets/oss/fbanime-gan/g_synthesis.onnx"
    };
    
    // Check if models exist
    bool models_exist = true;
    for (const auto& path : fbanime_paths)
    {
      if (!QFile::exists(QString::fromStdString(path)))
      {
        qDebug() << "Model not found:" << path.c_str();
        models_exist = false;
      }
    }
    
    if (models_exist)
    {
      auto fbanime_model = GANFactory::createModel("FBAnime", fbanime_paths);
      
      if (fbanime_model && fbanime_model->isReady())
      {
        qDebug() << "FBAnime model loaded successfully!";
        qDebug() << "Model type:" << fbanime_model->getModelType().c_str();
        qDebug() << "Latent size:" << fbanime_model->getLatentSize();
        auto [width, height] = fbanime_model->getOutputSize();
        qDebug() << "Output size:" << width << "x" << height;
        
        // Generate random image
        qDebug() << "\nGenerating random image...";
        auto start = std::chrono::high_resolution_clock::now();
        
        QImage random_image = fbanime_model->generateRandom();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "Generation took" << duration.count() << "ms";
        
        if (!random_image.isNull())
        {
          qDebug() << "Generated image size:" << random_image.width() << "x" << random_image.height();
          if (random_image.save("fbanime_random.png"))
          {
            qDebug() << "Saved as fbanime_random.png";
          }
        }
        else
        {
          qDebug() << "Failed to generate image";
        }
        
        // Test latent interpolation
        qDebug() << "\nTesting latent interpolation...";
        auto latent_a = GANUtils::generateRandomLatent(fbanime_model->getLatentSize());
        auto latent_b = GANUtils::generateRandomLatent(fbanime_model->getLatentSize());
        
        for (int i = 0; i <= 4; ++i)
        {
          float t = i / 4.0f;
          auto interpolated = GANUtils::interpolateLatents(latent_a, latent_b, t);
          auto image = fbanime_model->generateFromLatent(interpolated);
          
          if (!image.isNull())
          {
            QString filename = QString("fbanime_interp_%1.png").arg(i);
            image.save(filename);
            qDebug() << "Saved interpolation step" << i << "as" << filename;
          }
        }
      }
      else
      {
        qDebug() << "Failed to load FBAnime model";
      }
    }
    else
    {
      qDebug() << "FBAnime models not available, skipping test";
    }
#endif
    // Test other models from PINTO zoo if available
    qDebug() << "\n=== Testing PINTO Model Zoo GANs ===";
    
    // Test EigenGAN model (temporarily disabled due to ONNX conversion issues)
    qDebug() << "\nEigenGAN test temporarily disabled due to input name issues with TensorFlow->ONNX conversion";
    
    // Test MobileStyleGAN
    std::string mobile_stylegan_path
        = "/home/jcelerier/projets/oss/MobileStyleGAN.pytorch/"
          "SynthesisNetwork.onnx";
    if (QFile::exists(QString::fromStdString(mobile_stylegan_path)))
    {
      qDebug() << "\nTesting MobileStyleGAN model...";
      std::vector<std::string> mobile_paths = {mobile_stylegan_path};
      
      auto mobile_model = GANFactory::createModel("MobileStyleGAN", mobile_paths);
      
      if (mobile_model && mobile_model->isReady())
      {
        qDebug() << "MobileStyleGAN model loaded successfully!";
        qDebug() << "Model type:" << mobile_model->getModelType().c_str();
        qDebug() << "Latent size:" << mobile_model->getLatentSize();
        auto [width, height] = mobile_model->getOutputSize();
        qDebug() << "Output size:" << width << "x" << height;
        
        // Generate random image
        auto start = std::chrono::high_resolution_clock::now();
        QImage mobile_image = mobile_model->generateRandom();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "MobileStyleGAN generation took" << duration.count() << "ms";
        qDebug() << "Generated image dimensions:" << mobile_image.width() << "x" << mobile_image.height();
        
        if (!mobile_image.isNull())
        {
          mobile_image.save("mobilestylegan_random.png");
          qDebug() << "Saved MobileStyleGAN output as mobilestylegan_random.png";
        }
        else
        {
          qDebug() << "Failed to generate MobileStyleGAN image";
        }
      }
      else
      {
        qDebug() << "Failed to load MobileStyleGAN model";
      }
    }
    else
    {
      qDebug() << "MobileStyleGAN model not found at" << mobile_stylegan_path.c_str();
    }

#if 0
    // Test PyTorchGAN model
    std::string pytorch_gan_path = "/home/jcelerier/projets/oss/ailia-models/generative_adversarial_networks/pytorch-gan/pytorch-gnet-celeba.onnx";
    if (QFile::exists(QString::fromStdString(pytorch_gan_path)))
    {
      qDebug() << "\nTesting PyTorchGAN model...";
      std::vector<std::string> pytorch_paths = {pytorch_gan_path};
      
      auto pytorch_model = GANFactory::createModel("PyTorchGAN", pytorch_paths);
      
      if (pytorch_model && pytorch_model->isReady())
      {
        qDebug() << "PyTorchGAN model loaded successfully!";
        qDebug() << "Model type:" << pytorch_model->getModelType().c_str();
        qDebug() << "Latent size:" << pytorch_model->getLatentSize();
        auto [width, height] = pytorch_model->getOutputSize();
        qDebug() << "Output size:" << width << "x" << height;
        
        // Generate random image
        auto start = std::chrono::high_resolution_clock::now();
        QImage pytorch_image = pytorch_model->generateRandom();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "PyTorchGAN generation took" << duration.count() << "ms";
        
        if (!pytorch_image.isNull())
        {
          pytorch_image.save("pytorch_gan_random.png");
          qDebug() << "Saved PyTorchGAN output as pytorch_gan_random.png";
          
          // Test interpolation
          qDebug() << "Testing PyTorchGAN latent interpolation...";
          auto latent_a = GANUtils::generateUniformRandom(pytorch_model->getLatentSize());
          auto latent_b = GANUtils::generateUniformRandom(pytorch_model->getLatentSize());
          
          for (int i = 0; i <= 2; ++i)
          {
            float t = i / 2.0f;
            auto interpolated = GANUtils::interpolateLatents(latent_a, latent_b, t);
            auto image = pytorch_model->generateFromLatent(interpolated);
            
            if (!image.isNull())
            {
              QString filename = QString("pytorch_gan_interp_%1.png").arg(i);
              image.save(filename);
              qDebug() << "Saved PyTorchGAN interpolation step" << i << "as" << filename;
            }
          }
        }
        else
        {
          qDebug() << "Failed to generate PyTorchGAN image";
        }
      }
      else
      {
        qDebug() << "Failed to load PyTorchGAN model";
      }
    }
    else
    {
      qDebug() << "PyTorchGAN model not found at" << pytorch_gan_path.c_str();
    }
#endif

    // Test AnimeGANv3 Image Translation model
    std::string animegan_path = "/home/jcelerier/projets/oss/ailia-models/generative_adversarial_networks/pytorch-gan/AnimeGANv3_Hayao_36.onnx";
    if (QFile::exists(QString::fromStdString(animegan_path)))
    {
      qDebug() << "\nTesting AnimeGANv3 image translation model...";
      std::vector<std::string> animegan_paths = {animegan_path};
      
      auto animegan_model = GANFactory::createModel("AnimeGANv3", animegan_paths);
      
      if (animegan_model && animegan_model->isReady())
      {
        qDebug() << "AnimeGANv3 model loaded successfully!";
        qDebug() << "Model type:" << animegan_model->getModelType().c_str();
        qDebug() << "Is generative model:" << (animegan_model->isGenerativeModel() ? "Yes" : "No");
        qDebug() << "Latent size:" << animegan_model->getLatentSize();
        auto [width, height] = animegan_model->getOutputSize();
        qDebug() << "Output size:" << width << "x" << height;
        
        // Create test input image (gradient pattern)
        QImage test_image(256, 256, QImage::Format_RGB888);
        for (int y = 0; y < 256; ++y) {
          for (int x = 0; x < 256; ++x) {
            int r = (x * 255) / 256;      // Red gradient left-right
            int g = (y * 255) / 256;      // Green gradient top-bottom
            int b = 128;                  // Blue constant
            test_image.setPixel(x, y, qRgb(r, g, b));
          }
        }
        
        // Test image transformation
        auto start = std::chrono::high_resolution_clock::now();
        QImage transformed_image = animegan_model->transformImage(test_image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "Image transformation took" << duration.count() << "ms";
        
        if (!transformed_image.isNull())
        {
          qDebug() << "Generated transformed image size:" << transformed_image.width() << "x" << transformed_image.height();
          if (transformed_image.save("animegan_v3_transform.png"))
          {
            qDebug() << "Saved transformed image as animegan_v3_transform.png";
          }
          
          // Test with existing sitting.jpg if available
          if (QFile::exists("sitting.jpg"))
          {
            QImage real_image("sitting.jpg");
            if (!real_image.isNull())
            {
              qDebug() << "Testing with real image (sitting.jpg)...";
              auto real_transformed = animegan_model->transformImage(real_image);
              if (!real_transformed.isNull() && real_transformed.save("animegan_v3_real_transform.png"))
              {
                qDebug() << "Saved real image transformation as animegan_v3_real_transform.png";
              }
            }
          }
        }
        else
        {
          qDebug() << "Failed to transform image";
        }
      }
      else
      {
        qDebug() << "Failed to load AnimeGANv3 model";
      }
    }
    else
    {
      qDebug() << "AnimeGANv3 model not found at" << animegan_path.c_str();
    }
    
    qDebug() << "\n=== Known Issues ===";
    qDebug() << "PyTorchGAN: Model contains incompatible ONNX export with dynamic batch dimensions";
    qDebug() << "            that cannot be resolved by ONNX Runtime C++ (works in Python).";
    qDebug() << "EigenGAN:   Disabled due to empty input names from TensorFlow->ONNX conversion.";
    qDebug() << "MobileStyleGAN: Works but produces noisy output (expected per paper analysis).";
    qDebug() << "AnimeGANv3: Image-to-image translation model for real-time anime stylization.";
    
    qDebug() << "\n=== Generic GAN Architecture Test Complete ===";
    
    return 0;
  }
  catch (const Ort::Exception& e)
  {
    qDebug() << "ONNX Runtime error:" << e.what();
    return 1;
  }
  catch (const std::exception& e)
  {
    qDebug() << "Error:" << e.what();
    return 1;
  }
}
