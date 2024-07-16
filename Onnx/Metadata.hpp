#pragma once
#include <Process/ProcessMetadata.hpp>

namespace Onnx
{
class Model;
}

PROCESS_METADATA(
    ,
    Onnx::Model,
    "3b2e1535-51ad-444f-b095-a1c0cc39538c",
    "Onnx",                                  // Internal name
    "Onnx",                                  // Pretty name
    Process::ProcessCategory::Other,              // Category
    "Other",                                      // Category
    "Description",                                // Description
    "Author",                                     // Author
    (QStringList{"Put", "Your", "Tags", "Here"}), // Tags
    {},                                           // Inputs
    {},                                           // Outputs
    Process::ProcessFlags::SupportsAll            // Flags
)
