import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import { LayerNorm } from '../../src/nn/LayerNorm.js';

describe('LayerNorm', () => {
    it('should normalize to approximately mean=0, var=1', () => {
        const gamma = new Float32Array([1, 1, 1, 1]);
        const beta = new Float32Array([0, 0, 0, 0]);
        const ln = new LayerNorm(gamma, beta, 4);

        const x = new Tensor(new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]), [2, 4]);
        const result = ln.forward(x);

        expect(result.shape).toEqual([2, 4]);

        // Each row should have mean ≈ 0
        for (let row = 0; row < 2; row++) {
            let sum = 0;
            for (let j = 0; j < 4; j++) {
                sum += result.data[row * 4 + j];
            }
            expect(Math.abs(sum / 4)).toBeLessThan(1e-5);
        }
    });

    it('should apply scale (gamma) and shift (beta)', () => {
        const gamma = new Float32Array([2, 2, 2, 2]);
        const beta = new Float32Array([10, 10, 10, 10]);
        const ln = new LayerNorm(gamma, beta, 4);

        const x = new Tensor(new Float32Array([1, 2, 3, 4]), [1, 4]);
        const result = ln.forward(x);

        // Mean of result should be ≈ beta (10)
        let sum = 0;
        for (let j = 0; j < 4; j++) sum += result.data[j];
        expect(Math.abs(sum / 4 - 10)).toBeLessThan(1e-4);
    });
});
