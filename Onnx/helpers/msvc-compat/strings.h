#pragma once
// MSVC lacks <strings.h>. ctx.h includes it unconditionally but uses no
// strings.h function, so mapping to <string.h> is sufficient.
#include <string.h>
