/**
 * @file src/nn/Embedding.js
 * @description Embedding lookup table. Forward-only.
 */

import { Tensor } from '../core/Tensor.js';
import { ShapeError, ValidationError } from '../errors.js';

class Embedding {
    /**
     * @param {Float32Array} weightsData - Flat weight data [vocabSize * embeddingDim].
     * @param {number} vocabSize
     * @param {number} embeddingDim
     */
    constructor(weightsData, vocabSize, embeddingDim) {
        this.weights = new Tensor(weightsData, [vocabSize, embeddingDim]);
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
    }

    /**
     * Look up embeddings for token IDs.
     * @param {Tensor} ids - Shape [seqLen] with integer token IDs.
     * @returns {Tensor} Shape [seqLen, embeddingDim]
     * @throws {ShapeError} If ids is not 1D.
     * @throws {ValidationError} If a token ID is out of range.
     */
    forward(ids) {
        if (ids.shape.length !== 1) {
            throw new ShapeError('Embedding.forward expects a 1D tensor of token IDs.', {
                expected: 'seqLen',
                actual: `${ids.shape}`,
                operation: 'Embedding.forward',
            });
        }

        const seqLen = ids.shape[0];
        const result = new Float32Array(seqLen * this.embeddingDim);

        for (let i = 0; i < seqLen; i++) {
            const rawId = Math.round(ids.data[i]);
            const id = Math.max(0, Math.min(this.vocabSize - 1, rawId));
            if (rawId < 0 || rawId >= this.vocabSize) {
                throw new ValidationError(
                    `Token ID ${id} is out of range [0, ${this.vocabSize - 1}].`,
                    { tokenIndex: i, tokenId: id, vocabSize: this.vocabSize }
                );
            }
            const srcOffset = id * this.embeddingDim;
            const dstOffset = i * this.embeddingDim;
            for (let j = 0; j < this.embeddingDim; j++) {
                result[dstOffset + j] = this.weights.data[srcOffset + j];
            }
        }

        return new Tensor(result, [seqLen, this.embeddingDim]);
    }
}

export { Embedding };
