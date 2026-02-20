import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import '../../src/core/Ops.js'; // registers .dot/.add/.softmax on Tensor.prototype
import { Linear } from '../../src/nn/Linear.js';
import { MultiHeadAttention } from '../../src/nn/Attention.js';

function makeIdentityLinear(dim) {
    // Identity weights: [dim, dim] diagonal
    const w = new Float32Array(dim * dim);
    for (let i = 0; i < dim; i++) w[i * dim + i] = 1;
    return new Linear(w, null, dim, dim);
}

describe('MultiHeadAttention', () => {
    it('should produce output with correct shape', () => {
        const dim = 4;
        const numHeads = 2;
        const headDim = 2;

        const wq = makeIdentityLinear(dim);
        const wk = makeIdentityLinear(dim);
        const wv = makeIdentityLinear(dim);
        const wo = makeIdentityLinear(dim);

        const attn = new MultiHeadAttention(wq, wk, wv, wo, numHeads, headDim);
        const x = new Tensor(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]), [3, 4]);
        const result = attn.forward(x);

        expect(result.shape).toEqual([3, 4]);
    });

    it('should apply causal masking (first token sees only itself)', () => {
        const dim = 2;
        const numHeads = 1;
        const headDim = 2;

        const wq = makeIdentityLinear(dim);
        const wk = makeIdentityLinear(dim);
        const wv = makeIdentityLinear(dim);
        const wo = makeIdentityLinear(dim);

        const attn = new MultiHeadAttention(wq, wk, wv, wo, numHeads, headDim);

        // Two tokens: [1, 0] and [0, 1]
        const x = new Tensor(new Float32Array([1, 0, 0, 1]), [2, 2]);
        const result = attn.forward(x);

        // First token's output should be based only on itself due to causal mask
        // With identity projections, first token attends only to itself → output ≈ [1, 0]
        expect(Math.abs(result.data[0] - 1)).toBeLessThan(0.1);
        expect(Math.abs(result.data[1] - 0)).toBeLessThan(0.1);
    });

    it('should work with single token (seqLen=1)', () => {
        const dim = 4;
        const numHeads = 2;
        const headDim = 2;

        const wq = makeIdentityLinear(dim);
        const wk = makeIdentityLinear(dim);
        const wv = makeIdentityLinear(dim);
        const wo = makeIdentityLinear(dim);

        const attn = new MultiHeadAttention(wq, wk, wv, wo, numHeads, headDim);
        const x = new Tensor(new Float32Array([1, 2, 3, 4]), [1, 4]);
        const result = attn.forward(x);

        expect(result.shape).toEqual([1, 4]);
        // Single token sees only itself → output should be close to input with identity weights
        for (let i = 0; i < 4; i++) {
            expect(Math.abs(result.data[i] - x.data[i])).toBeLessThan(0.1);
        }
    });

    it('should throw if embedDim % numHeads != 0', () => {
        const dim = 5;
        const numHeads = 2;
        const headDim = 2; // 2*2=4 ≠ 5

        const wq = makeIdentityLinear(dim);
        const wk = makeIdentityLinear(dim);
        const wv = makeIdentityLinear(dim);
        const wo = makeIdentityLinear(dim);

        expect(
            () => new MultiHeadAttention(wq, wk, wv, wo, numHeads, headDim)
        ).toThrow('divisible');
    });
});
