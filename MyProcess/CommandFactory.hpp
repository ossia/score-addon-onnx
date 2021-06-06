#pragma once
#include <score/command/Command.hpp>

namespace MyProcess
{
inline const CommandGroupKey& CommandFactoryName()
{
  static const CommandGroupKey key{"MyProcess"};
  return key;
}
}
