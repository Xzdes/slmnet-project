import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import { Embedding } from '../../src/nn/Embedding.js';

describe('Embedding', () => {
    const vocabSize = 5;
    const embedDim = 3;
    // Simple weight: each token i maps to [i, i, i]
    const weights = new Float32Array([
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4, 4,
    ]);

    it('should look up embeddings for 1D input', () => {
        const emb = new Embedding(weights, vocabSize, embedDim);
        const ids = new Tensor(new Float32Array([0, 2, 4]), [3]);
        const out = emb.forward(ids);
        expect(out.shape).toEqual([3, 3]);
        expect(Array.from(out.data)).toEqual([0, 0, 0, 2, 2, 2, 4, 4, 4]);
    });

    it('should throw on 2D input', () => {
        const emb = new Embedding(weights, vocabSize, embedDim);
        const ids = new Tensor(new Float32Array([0, 1, 2, 3]), [2, 2]);
        expect(() => emb.forward(ids)).toThrow('1D tensor');
    });

    it('should throw on out-of-range token ID', () => {
        const emb = new Embedding(weights, vocabSize, embedDim);
        const ids = new Tensor(new Float32Array([0, 10]), [2]);
        expect(() => emb.forward(ids)).toThrow('out of range');
    });

    it('should handle float token IDs by rounding', () => {
        const emb = new Embedding(weights, vocabSize, embedDim);
        const ids = new Tensor(new Float32Array([1.9, 3.1]), [2]);
        const out = emb.forward(ids);
        // 1.9 rounds to 2, 3.1 rounds to 3
        expect(Array.from(out.data)).toEqual([2, 2, 2, 3, 3, 3]);
    });
});
