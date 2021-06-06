#pragma once
#include <Process/ProcessMetadata.hpp>

namespace MyProcess
{
class Model;
}

PROCESS_METADATA(
    ,
    MyProcess::Model,
    "00000000-0000-0000-0000-000000000000",
    "MyProcess",                                  // Internal name
    "MyProcess",                                  // Pretty name
    Process::ProcessCategory::Other,              // Category
    "Other",                                      // Category
    "Description",                                // Description
    "Author",                                     // Author
    (QStringList{"Put", "Your", "Tags", "Here"}), // Tags
    {},                                           // Inputs
    {},                                           // Outputs
    Process::ProcessFlags::SupportsAll            // Flags
)
