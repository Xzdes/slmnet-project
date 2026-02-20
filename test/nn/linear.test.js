import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import { Ops } from '../../src/core/Ops.js';
import { Linear } from '../../src/nn/Linear.js';

describe('Linear', () => {
    it('should compute forward pass with bias', () => {
        // W: [2,3], b: [1,3], x: [1,2]
        const w = new Float32Array([1, 0, 0, 0, 1, 0]); // identity-like
        const b = new Float32Array([10, 20, 30]);
        const layer = new Linear(w, b, 2, 3);

        const x = new Tensor(new Float32Array([5, 7]), [1, 2]);
        const out = layer.forward(x);
        expect(out.shape).toEqual([1, 3]);
        expect(out.data[0]).toBe(5 + 10);
        expect(out.data[1]).toBe(7 + 20);
        expect(out.data[2]).toBe(0 + 30);
    });

    it('should compute forward pass without bias', () => {
        const w = new Float32Array([1, 2, 3, 4]);
        const layer = new Linear(w, null, 2, 2);

        const x = new Tensor(new Float32Array([1, 1]), [1, 2]);
        const out = layer.forward(x);
        expect(out.shape).toEqual([1, 2]);
        // [1,1] x [[1,2],[3,4]] = [4, 6]
        expect(out.data[0]).toBe(4);
        expect(out.data[1]).toBe(6);
    });

    it('should handle batch input', () => {
        const w = new Float32Array([1, 0, 0, 1]);
        const layer = new Linear(w, null, 2, 2);

        const x = Tensor.from([[1, 2], [3, 4]]);
        const out = layer.forward(x);
        expect(out.shape).toEqual([2, 2]);
    });
});
