import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import { Sampler } from '../../src/runtime/Sampler.js';

describe('Sampler', () => {
    it('should return a valid token ID', () => {
        const logits = Tensor.zeros([10]);
        const id = Sampler.sample(logits);
        expect(id).toBeGreaterThanOrEqual(0);
        expect(id).toBeLessThan(10);
    });

    it('with topK=1, should always pick the max', () => {
        const logits = new Tensor(new Float32Array([1, 5, 2]), [3]);
        const id = Sampler.sample(logits, { topK: 1, temperature: 1.0 });
        expect(id).toBe(1); // index of value 5
    });

    it('with very low temperature, should gravitate to max', () => {
        const logits = new Tensor(new Float32Array([0, 0, 10, 0, 0]), [5]);
        // Run multiple times — with temp=0.01, should almost always pick index 2
        let count = 0;
        for (let i = 0; i < 20; i++) {
            if (Sampler.sample(logits, { temperature: 0.01 }) === 2) count++;
        }
        expect(count).toBeGreaterThan(15);
    });

    it('should respect topP filtering', () => {
        // Create logits where one token dominates
        const logits = new Tensor(new Float32Array([10, -10, -10, -10, -10]), [5]);
        const id = Sampler.sample(logits, { topP: 0.5, temperature: 1.0 });
        expect(id).toBe(0); // Only token 0 passes the topP threshold
    });
});
