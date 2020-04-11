/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 * =============================================================================
 */

import {KernelConfig, Tensor, Transpose, TransposeAttrs, TransposeInputs, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {TransposeSharedProgram} from './transpose_shared_webgpu';
import {TransposeProgram} from './transpose_webgpu';

export const transposeConfig: KernelConfig = {
  kernelName: Transpose,
  backendName: 'webgpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as TransposeInputs;
    const {perm} = attrs as {} as TransposeAttrs;
    const webGPUBackend = backend as WebGPUBackend;

    const xRank = x.shape.length;

    const newShape: number[] = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }

    if (x.shape.length === 2 && util.arraysEqual(perm, [1, 0])) {
      const program = new TransposeSharedProgram(x.shape, perm);
      return webGPUBackend.compileAndRun(program, [x as Tensor]);
    }
    const program = new TransposeProgram(x.shape, perm);
    return webGPUBackend.compileAndRun(program, [x as Tensor]);
  }
};
