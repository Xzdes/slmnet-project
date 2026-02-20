import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import '../../src/core/Ops.js'; // registers .dot/.add/.softmax on Tensor.prototype
import { Linear } from '../../src/nn/Linear.js';
import { LayerNorm } from '../../src/nn/LayerNorm.js';
import { MultiHeadAttention } from '../../src/nn/Attention.js';
import { FeedForward } from '../../src/nn/FeedForward.js';
import { TransformerBlock } from '../../src/nn/TransformerBlock.js';

function makeIdentityLinear(dim) {
    const w = new Float32Array(dim * dim);
    for (let i = 0; i < dim; i++) w[i * dim + i] = 1;
    return new Linear(w, null, dim, dim);
}

function makeLayerNorm(dim) {
    return new LayerNorm(
        new Float32Array(dim).fill(1),
        new Float32Array(dim).fill(0),
        dim
    );
}

describe('TransformerBlock', () => {
    it('should produce output with same shape as input (residual)', () => {
        const dim = 4;
        const numHeads = 2;
        const headDim = 2;
        const hiddenDim = 8;

        const attn = new MultiHeadAttention(
            makeIdentityLinear(dim),
            makeIdentityLinear(dim),
            makeIdentityLinear(dim),
            makeIdentityLinear(dim),
            numHeads,
            headDim
        );

        const w1 = new Float32Array(dim * hiddenDim).fill(0.1);
        const w2 = new Float32Array(hiddenDim * dim).fill(0.1);
        const ffn = new FeedForward(
            new Linear(w1, null, dim, hiddenDim),
            new Linear(w2, null, hiddenDim, dim)
        );

        const block = new TransformerBlock(attn, ffn, makeLayerNorm(dim), makeLayerNorm(dim));

        const x = new Tensor(
            new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            [3, 4]
        );
        const result = block.forward(x);

        expect(result.shape).toEqual([3, 4]);
    });

    it('should have residual connections (output ≠ 0 when input ≠ 0)', () => {
        const dim = 4;
        const numHeads = 2;
        const headDim = 2;

        // Zero all weights in projections → attention output is ~0
        const zeroLinear = new Linear(
            new Float32Array(dim * dim).fill(0),
            null,
            dim,
            dim
        );

        const attn = new MultiHeadAttention(
            zeroLinear,
            makeIdentityLinear(dim),
            makeIdentityLinear(dim),
            makeIdentityLinear(dim),
            numHeads,
            headDim
        );

        // Zero FFN too
        const zeroFfn = new FeedForward(
            new Linear(new Float32Array(dim * dim).fill(0), null, dim, dim),
            new Linear(new Float32Array(dim * dim).fill(0), null, dim, dim)
        );

        const block = new TransformerBlock(
            attn,
            zeroFfn,
            makeLayerNorm(dim),
            makeLayerNorm(dim)
        );

        const x = new Tensor(new Float32Array([1, 2, 3, 4]), [1, 4]);
        const result = block.forward(x);

        // With residual connections, even if sublayers output ~0, result ≈ input
        let norm = 0;
        for (let i = 0; i < 4; i++) norm += result.data[i] * result.data[i];
        expect(norm).toBeGreaterThan(0.1);
    });
});
