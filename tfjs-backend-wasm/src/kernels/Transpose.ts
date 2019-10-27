/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {backend_util, buffer, NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface TransposeInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface TransposeAttrs extends NamedAttrMap {
  perm: number[];
}

let wasmTranspose: (
    xId: number, xShape: Uint8Array, xShapeLength: number, outId: number,
    outShape: Uint8Array, outShapeLength: number, perm: Uint8Array,
    permLength: number) => void;

function setup(backend: BackendWasm) {
  wasmTranspose = backend.wasm.cwrap('Transpose', null /* void */, [
    'number',  // xId
    'array',   // x.shape
    'number',  // x.shape.length
    'number',  // outId
    'array',   // out.shape
    'number',  // out.shape.length
    'array',   // perm
    'number',  // perm.length
  ]);
}

function transpose(
    args:
        {inputs: TransposeInputs, backend: BackendWasm, attrs: TransposeAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const [reducedShape, perm] = removeOneSizeDims(inputs.x.shape, attrs.perm);
  const x = {
    dataId: inputs.x.dataId,
    shape: reducedShape,
    dtype: inputs.x.dtype
  };
  const rank = x.shape.length;
  let noChange = true;
  for (let i = 0; i < perm.length; i++) {
    if (perm[i] !== i) {
      noChange = false;
    }
  }
  const outShape = computeOutShape(inputs.x.shape, attrs.perm);
  if (noChange) {
    return {dataId: x.dataId, shape: outShape, dtype: x.dtype};
  }
  const out = backend.makeOutput(outShape, x.dtype);
  if (rank <= 3) {
    const xId = backend.dataIdMap.get(x.dataId).id;
    const outId = backend.dataIdMap.get(out.dataId).id;
    const permBytes = new Uint8Array(new Int32Array(perm).buffer);
    const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
    const outShapeBytes = new Uint8Array(new Int32Array(out.shape).buffer);
    wasmTranspose(
        xId, xShapeBytes, x.shape.length, outId, outShapeBytes,
        out.shape.length, permBytes, perm.length);
  } else {
    const xVals = backend.typedArrayFromHeap(x);
    const outVals = backend.typedArrayFromHeap(out);
    genericSlowTranspose(xVals, x, outVals, out, perm);
  }
  return out;
}

function genericSlowTranspose(
    xVals: backend_util.TypedArray, xInfo: TensorInfo,
    outVals: backend_util.TypedArray, outInfo: TensorInfo,
    perm: number[]): void {
  const xBuf = buffer(xInfo.shape, xInfo.dtype, xVals);
  const outBuf = buffer(outInfo.shape, outInfo.dtype, outVals);
  for (let i = 0; i < xBuf.size; ++i) {
    const loc = xBuf.indexToLoc(i);
    // Permute location.
    const newLoc: number[] = new Array(loc.length);
    for (let i = 0; i < newLoc.length; i++) {
      newLoc[i] = loc[perm[i]];
    }
    const newIndex = outBuf.locToIndex(newLoc);
    outVals[newIndex] = xVals[i];
  }
}

function computeOutShape(inShape: number[], perm: number[]): number[] {
  const outShape = new Array(inShape.length);
  for (let i = 0; i < outShape.length; i++) {
    outShape[i] = inShape[perm[i]];
  }
  return outShape;
}

function removeOneSizeDims(
    shape: number[], perm: number[]): [number[], number[]] {
  const newShape: number[] = [];
  const newPerm: number[] = [];
  for (let i = 0; i < shape.length; ++i) {
    if (shape[i] !== 1) {
      newShape.push(shape[i]);
    }
    if (shape[perm[i]] !== 1) {
      newPerm.push(perm[i]);
    }
  }
  for (let i = 0; i < newPerm.length; ++i) {
    let minValIdx = -1;
    for (let j = 0; j < newPerm.length; ++j) {
      if (newPerm[j] >= i &&
          (minValIdx === -1 || newPerm[minValIdx] > newPerm[j])) {
        minValIdx = j;
      }
    }
    newPerm[minValIdx] = i;
  }
  return [newShape, newPerm];
}

registerKernel({
  kernelName: 'Transpose',
  backendName: 'wasm',
  kernelFunc: transpose,
  setupFunc: setup,
});
