import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import '../../src/core/Ops.js'; // registers .dot/.add/.gelu on Tensor.prototype
import { Linear } from '../../src/nn/Linear.js';
import { FeedForward } from '../../src/nn/FeedForward.js';

describe('FeedForward', () => {
    it('should produce output with correct shape', () => {
        const embedDim = 4;
        const hiddenDim = 8;

        // linear1: [embedDim, hiddenDim], linear2: [hiddenDim, embedDim]
        const w1 = new Float32Array(embedDim * hiddenDim).fill(0.1);
        const b1 = new Float32Array(hiddenDim).fill(0);
        const w2 = new Float32Array(hiddenDim * embedDim).fill(0.1);
        const b2 = new Float32Array(embedDim).fill(0);

        const linear1 = new Linear(w1, b1, embedDim, hiddenDim);
        const linear2 = new Linear(w2, b2, hiddenDim, embedDim);
        const ffn = new FeedForward(linear1, linear2);

        const x = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 4]);
        const result = ffn.forward(x);

        expect(result.shape).toEqual([2, 4]);
    });

    it('should apply GELU nonlinearity (output != linear)', () => {
        const dim = 2;
        const hidden = 2;

        // Identity-like weights
        const w1 = new Float32Array([1, 0, 0, 1]);
        const w2 = new Float32Array([1, 0, 0, 1]);
        const linear1 = new Linear(w1, null, dim, hidden);
        const linear2 = new Linear(w2, null, hidden, dim);
        const ffn = new FeedForward(linear1, linear2);

        // Negative input: GELU(-5) ≈ 0, so output should be close to 0 for negative values
        const x = new Tensor(new Float32Array([-5, -5]), [1, 2]);
        const result = ffn.forward(x);

        // GELU squashes large negatives to ~0
        expect(Math.abs(result.data[0])).toBeLessThan(0.1);
    });
});
