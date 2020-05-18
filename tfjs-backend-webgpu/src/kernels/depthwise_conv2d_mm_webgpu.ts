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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {computeDispatch} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation, inDims;';
  workGroupSize: [number, number, number] = [4, 16, 1];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    const channelMul = convInfo.outChannels / convInfo.inChannels;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');

    this.userCode = `
    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];  //Ci * Mul
    int dimInner = xShape[3] * filterDims[0] * filterDims[1];

    float mm_readA(int row, int col) {
      int r = int(row), c = int(col);
      int outRow = r / outShape[2];
      int outCol = r % outShape[2];
      int WRow = (c / filterDims[1]) % filterDims[0];
      int WCol = c % filterDims[1];

      int Ci = c / (filterDims[0] * filterDims[1]);

      ivec4 coord = ivec4(
          batch,
          outRow * stride[0] + WRow * dilation[0] - pad[0],
          outCol * stride[1] + WCol * dilation[1] - pad[1],
          Ci);
      return coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0;
    }

    float mm_readB(int row, int col) {
      int r = int(row), c= int(col);
      int Ci =  r / (filterDims[0] * filterDims[1]);

      if(Ci == int(c / ${channelMul})) {
        int WRow = (r / (filterDims[1]) % filterDims[0]);
        int WCol = r % filterDims[1];
        int Mul = c % ${channelMul};
        ivec4 coord = ivec4(WRow, WCol, Ci, Mul);
        return coordsInBounds(coord, wShape) ? W[getFlatIndex(coord, wShape)] : 0;
      }

      return 0;
    }

    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / outShape[2],
          row % outShape[2],
          col);
      result[getFlatIndex(outCoord, outShape)] = value;
      }

    const int MatTileSize = int(gl_WorkGroupSize.x);  // .x == .y
    shared float mm_Asub[MatTileSize][MatTileSize];
    shared float mm_Bsub[MatTileSize][MatTileSize];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
        int localRow = int(gl_LocalInvocationID.y);  // 0..MatTileSize
        int localCol = int(gl_LocalInvocationID.x);  // 0..MatTileSize
        int globalRow = int(gl_GlobalInvocationID.y);  // AOuter
        int globalCol = int(gl_GlobalInvocationID.x);  // Inner

        float acc = 0.0;
        int numTiles = (dimInner - 1) / MatTileSize + 1;

        for (int t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          int tiledACol = MatTileSize * t + localCol;
          int tiledBRow = MatTileSize * t + localRow;
          mm_Asub[localRow][localCol] = mm_readA(globalRow, tiledACol);
          mm_Bsub[localRow][localCol] = mm_readB(tiledBRow, globalCol);

          // Synchronise to make sure the tile is loaded
          barrier();

          for (int k = 0; k < MatTileSize; k++) {
            acc += mm_Asub[localRow][k] * mm_Bsub[k][localCol];
          }

          // Synchronise before loading the next tile
          barrier();
        }

        if (globalCol < dimBOuter && globalRow < dimAOuter) {
          mm_write(globalRow, globalCol, acc);
        }
    }

    void main() {
      batch = int(gl_GlobalInvocationID.z);
      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
    `;

    this.shaderKey = `depthwise${channelMul}`;
  }
}
