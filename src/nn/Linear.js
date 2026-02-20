/**
 * @file src/nn/Linear.js
 * @description Fully-connected (dense) layer. Forward-only.
 */

import { Tensor } from '../core/Tensor.js';

class Linear {
    /**
     * @param {Float32Array} weightsData - Flat weight data [inFeatures * outFeatures].
     * @param {Float32Array|null} biasData - Flat bias data [outFeatures], or null.
     * @param {number} inFeatures
     * @param {number} outFeatures
     */
    constructor(weightsData, biasData, inFeatures, outFeatures) {
        this.weights = new Tensor(weightsData, [inFeatures, outFeatures]);
        this.bias = biasData ? new Tensor(biasData, [1, outFeatures]) : null;
    }

    /**
     * @param {Tensor} x - Shape [rows, inFeatures]
     * @returns {Tensor} Shape [rows, outFeatures]
     */
    forward(x) {
        let out = x.dot(this.weights);
        if (this.bias) out = out.add(this.bias);
        return out;
    }
}

export { Linear };
