/**
 * @file src/nn/LayerNorm.js
 * @description Layer Normalization. Forward-only.
 */

import { Tensor } from '../core/Tensor.js';
import { Ops } from '../core/Ops.js';

class LayerNorm {
    /**
     * @param {Float32Array} gammaData - Scale parameters [featureDim].
     * @param {Float32Array} betaData - Shift parameters [featureDim].
     * @param {number} featureDim
     * @param {number} [eps=1e-5]
     */
    constructor(gammaData, betaData, featureDim, eps = 1e-5) {
        this.gamma = new Tensor(gammaData, [featureDim]);
        this.beta = new Tensor(betaData, [featureDim]);
        this.eps = eps;
    }

    /**
     * @param {Tensor} x - Shape [rows, featureDim]
     * @returns {Tensor}
     */
    forward(x) {
        return Ops.layerNorm(x, this.gamma, this.beta, this.eps);
    }
}

export { LayerNorm };
