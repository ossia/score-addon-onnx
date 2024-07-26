#include "Process.hpp"
#include <Process/Dataflow/WidgetInlets.hpp>
#include <score/tools/File.hpp>
#include <wobjectimpl.h>

W_OBJECT_IMPL(Onnx::Model)

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This sample application demonstrates how to use components of the experimental C++ API
 * to query for model inputs/outputs and how to run inferrence on a model.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model should have float input
 *
 *
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) generate random data for an input tensor
 *  4) pass tensor through the model and check the resulting tensor
 *
 */

#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <string>
#include <vector>

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v)
{
    int total = 1;
    for (auto &i : v)
        total *= i;
    return total;
}

template<typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape)
{
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                          OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(mem_info,
                                              data.data(),
                                              data.size(),
                                              shape.data(),
                                              shape.size());
    return tensor;
}

struct ModelSpec
{
    struct Port
    {
        QString name;
        Process::PortType port_type{};
        ossia::val_type data_type{};
        std::vector<int64_t> shape;
    };
    std::vector<Port> inputs;
    std::vector<Port> outputs;

    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char;
    std::vector<const char *> output_names_char;
};

struct OnnxRunContext
{
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

    // print name/shape of inputs
    Ort::AllocatorWithDefaultOptions allocator;

    ModelSpec readModelSpec()
    {
        ModelSpec spec;

        for (std::size_t i = 0; i < session.GetInputCount(); i++) {
            const std::string name = session.GetInputNameAllocated(i, allocator).get();
            const Ort::TypeInfo &input_type = session.GetInputTypeInfo(i);
            const Ort::ConstTensorTypeAndShapeInfo &input_tensor_type
                = input_type.GetTensorTypeAndShapeInfo();

            spec.inputs.push_back({.name = QString::fromStdString(name),
                                   .port_type = {},
                                   .data_type = {},
                                   .shape = input_tensor_type.GetShape()});

            spec.input_names.push_back(std::move(name));
        }

        for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
            const std::string name = session.GetOutputNameAllocated(i, allocator).get();
            const Ort::TypeInfo &output_type = session.GetOutputTypeInfo(i);
            const Ort::ConstTensorTypeAndShapeInfo &output_tensor_type
                = output_type.GetTensorTypeAndShapeInfo();

            spec.outputs.push_back({.name = QString::fromStdString(name),
                                    .port_type = {},
                                    .data_type = {},
                                    .shape = output_tensor_type.GetShape()});

            spec.output_names.push_back(std::move(name));
        }
        spec.input_names_char.resize(spec.input_names.size());
        spec.output_names_char.resize(spec.output_names.size());
        std::transform(std::begin(spec.input_names),
                       std::end(spec.input_names),
                       std::begin(spec.input_names_char),
                       [&](const std::string &str) { return str.c_str(); });

        std::transform(std::begin(spec.output_names),
                       std::end(spec.output_names),
                       std::begin(spec.output_names_char),
                       [&](const std::string &str) { return str.c_str(); });

        // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
        // for (auto &s : input_shapes) {
        //     if (s < 0) {
        //         s = 1;
        //     }
        // }

        return spec;
    }

    explicit OnnxRunContext(QString name)
        : env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer")
        , session(env, name.toStdString().c_str(), session_options)
    {
        // Assume model has 1 input node and 1 output node.
        // assert(input_names.size() == 1 && output_names.size() == 1);
    }

    struct FloatTensor
    {
        std::vector<float> storage;
        Ort::Value value;
    };

    FloatTensor createRandomTensor(ModelSpec::Port &port)
    {
        // Create a single Ort tensor of random numbers
        auto &input_shape = port.shape;
        auto total_number_elements = calculate_product(input_shape);

        // generate random numbers in the range [0, 255]
        std::vector<float> input_tensor_values(total_number_elements);
        std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] {
            return rand() % 255;
        });

        return {.storage = std::move(input_tensor_values),
                .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
        /*
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back();

        // double-check the dimensions of the input tensor
        assert(input_tensors[0].IsTensor()
               && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);

        return input_tensors_values;
*/
    }

    void infer(const ModelSpec &spec, std::span<Ort::Value> input_tensors)
    {
        try {
            auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                              spec.input_names_char.data(),
                                              input_tensors.data(),
                                              spec.input_names_char.size(),
                                              spec.output_names_char.data(),
                                              spec.output_names_char.size());

            for (const Ort::Value &ot : output_tensors) {
                //    auto data = ot.GetTensorData<float>();
                std::cout << spec.output_names_char[0] << ": " << ot.GetCount() << std::endl;
            }

            // double-check the dimensions of the output tensors
            // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
            assert(output_tensors.size() == spec.output_names.size()
                   && output_tensors[0].IsTensor());
        } catch (const Ort::Exception &exception) {
            std::cout << "ERROR running model inference: " << exception.what() << std::endl;
            exit(-1);
        }
    }
};

namespace Onnx
{

Model::Model(const TimeVal &duration,
             const QString &name,
             const Id<Process::ProcessModel> &id,
             QObject *parent)
    : Process::ProcessModel{duration, id, "OnnxProcess", parent}
    , m_text{name}
{
  metadata().setInstanceName(*this);
  OnnxRunContext ctx(score::locateFilePath(name, score::IDocument::documentContext(*parent)));
  auto spec = ctx.readModelSpec();

  auto t = ctx.createRandomTensor(spec.inputs[0]);
  Ort::Value in[1]{std::move(t.value)};
  ctx.infer(spec, in);
}

Model::~Model() { }

QString Model::prettyName() const noexcept
{
  return tr("Onnx Process");
}
}
template <>
void DataStreamReader::read(const Onnx::Model& proc)
{
  insertDelimiter();
}

template <>
void DataStreamWriter::write(Onnx::Model& proc)
{
  checkDelimiter();
}

template <>
void JSONReader::read(const Onnx::Model& proc)
{
}

template <>
void JSONWriter::write(Onnx::Model& proc)
{
}
