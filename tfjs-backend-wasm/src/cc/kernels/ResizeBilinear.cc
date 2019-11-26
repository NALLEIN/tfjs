/* Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <vector>

#include <cmath>
#include "src/cc/backend.h"

#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void ResizeBilinear(int x_id, int batch, int old_height, int old_width,
                    int num_channels, int new_height, int new_width,
                    int align_corners, int out_id) {
  int effective_input_height =
      (align_corners > 0 && new_height > 1) ? old_height - 1 : old_height;
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
