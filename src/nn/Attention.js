/**
 * @file src/nn/Attention.js
 * @description Multi-Head Causal Self-Attention. Forward-only.
 */

import { Tensor } from '../core/Tensor.js';
import { ValidationError } from '../errors.js';

class MultiHeadAttention {
    /**
     * @param {Linear} wq - Query projection [embedDim, embedDim]
     * @param {Linear} wk - Key projection [embedDim, embedDim]
     * @param {Linear} wv - Value projection [embedDim, embedDim]
     * @param {Linear} wo - Output projection [embedDim, embedDim]
     * @param {number} numHeads
     * @param {number} headDim - embedDim / numHeads
     */
    constructor(wq, wk, wv, wo, numHeads, headDim) {
        if (headDim * numHeads !== wq.weights.shape[1]) {
            throw new ValidationError(
                `embedDim (${wq.weights.shape[1]}) must be divisible by numHeads (${numHeads}).`,
                { embedDim: wq.weights.shape[1], numHeads, headDim }
            );
        }
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.scale = 1 / Math.sqrt(headDim);
    }

    /**
     * @param {Tensor} x - Shape [seqLen, embedDim]
     * @returns {Tensor} Shape [seqLen, embedDim]
     */
    forward(x) {
        const [seqLen, embedDim] = x.shape;

        const Q = this.wq.forward(x); // [seqLen, embedDim]
        const K = this.wk.forward(x);
        const V = this.wv.forward(x);

        // Split into heads and compute attention per head
        const headOutputs = new Float32Array(seqLen * embedDim);

        for (let h = 0; h < this.numHeads; h++) {
            // Extract head slices
            const q = this._getHead(Q, h, seqLen);
            const k = this._getHead(K, h, seqLen);
            const v = this._getHead(V, h, seqLen);

            // Scaled dot-product attention: softmax(Q*K^T / sqrt(d)) * V
            const kT = k.transpose();
            const scores = q.dot(kT); // [seqLen, seqLen]

            // Apply causal mask + scale
            for (let i = 0; i < seqLen; i++) {
                for (let j = 0; j < seqLen; j++) {
                    const idx = i * seqLen + j;
                    if (j > i) {
                        scores.data[idx] = -1e9;
                    } else {
                        scores.data[idx] *= this.scale;
                    }
                }
            }

            const weights = scores.softmax(); // [seqLen, seqLen]
            const headOut = weights.dot(v); // [seqLen, headDim]

            // Write head output into combined buffer
            for (let i = 0; i < seqLen; i++) {
                for (let j = 0; j < this.headDim; j++) {
                    headOutputs[i * embedDim + h * this.headDim + j] =
                        headOut.data[i * this.headDim + j];
                }
            }
        }

        const combined = new Tensor(headOutputs, [seqLen, embedDim]);
        return this.wo.forward(combined);
    }

    /** @private */
    _getHead(tensor, headIdx, seqLen) {
        const embedDim = tensor.shape[1];
        const data = new Float32Array(seqLen * this.headDim);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < this.headDim; j++) {
                data[i * this.headDim + j] = tensor.data[i * embedDim + headIdx * this.headDim + j];
            }
        }
        return new Tensor(data, [seqLen, this.headDim]);
    }
}

export { MultiHeadAttention };
