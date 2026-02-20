import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import { Ops } from '../../src/core/Ops.js';

describe('Ops', () => {
    describe('matMul', () => {
        it('should multiply [2,3] x [3,2] correctly', () => {
            const a = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const b = new Tensor(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
            const c = Ops.matMul(a, b);
            expect(c.shape).toEqual([2, 2]);
            expect(Array.from(c.data)).toEqual([58, 64, 139, 154]);
        });

        it('should throw on incompatible shapes', () => {
            const a = Tensor.zeros([2, 3]);
            const b = Tensor.zeros([2, 3]);
            expect(() => Ops.matMul(a, b)).toThrow();
        });

        it('should throw on non-2D tensors', () => {
            const a = Tensor.zeros([3]);
            const b = Tensor.zeros([3, 2]);
            expect(() => Ops.matMul(a, b)).toThrow();
        });
    });

    describe('add', () => {
        it('should add same-shape tensors', () => {
            const a = Tensor.from([
                [1, 2],
                [3, 4],
            ]);
            const b = Tensor.from([
                [5, 6],
                [7, 8],
            ]);
            const c = Ops.add(a, b);
            expect(c.toArray()).toEqual([
                [6, 8],
                [10, 12],
            ]);
        });

        it('should broadcast [M,N] + [N]', () => {
            const a = Tensor.from([
                [1, 2],
                [3, 4],
            ]);
            const b = new Tensor(new Float32Array([10, 20]), [2]);
            const c = Ops.add(a, b);
            expect(c.toArray()).toEqual([
                [11, 22],
                [13, 24],
            ]);
        });

        it('should throw on incompatible shapes', () => {
            const a = Tensor.zeros([2, 3]);
            const b = Tensor.zeros([4]);
            expect(() => Ops.add(a, b)).toThrow();
        });
    });

    describe('mul', () => {
        it('should multiply same-shape tensors', () => {
            const a = Tensor.from([
                [1, 2],
                [3, 4],
            ]);
            const b = Tensor.from([
                [2, 3],
                [4, 5],
            ]);
            const c = Ops.mul(a, b);
            expect(c.toArray()).toEqual([
                [2, 6],
                [12, 20],
            ]);
        });

        it('should broadcast scalar', () => {
            const a = Tensor.from([
                [1, 2],
                [3, 4],
            ]);
            const s = new Tensor(new Float32Array([3]), [1]);
            const c = Ops.mul(a, s);
            expect(c.toArray()).toEqual([
                [3, 6],
                [9, 12],
            ]);
        });
    });

    describe('transpose', () => {
        it('should transpose [2,3] to [3,2]', () => {
            const a = Tensor.from([
                [1, 2, 3],
                [4, 5, 6],
            ]);
            const t = Ops.transpose(a);
            expect(t.shape).toEqual([3, 2]);
            expect(t.toArray()).toEqual([
                [1, 4],
                [2, 5],
                [3, 6],
            ]);
        });
    });

    describe('softmax', () => {
        it('should produce probabilities summing to 1 for 1D', () => {
            const t = new Tensor(new Float32Array([1, 2, 3]), [3]);
            const s = Ops.softmax(t);
            const sum = s.data.reduce((a, b) => a + b, 0);
            expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
        });

        it('should be numerically stable with large values', () => {
            const t = new Tensor(new Float32Array([1000, 1001, 1002]), [3]);
            const s = Ops.softmax(t);
            const sum = s.data.reduce((a, b) => a + b, 0);
            expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
        });

        it('should handle 2D row-wise softmax', () => {
            const t = Tensor.from([
                [1, 2],
                [3, 4],
            ]);
            const s = Ops.softmax(t);
            // Each row should sum to 1
            const row0 = s.data[0] + s.data[1];
            const row1 = s.data[2] + s.data[3];
            expect(Math.abs(row0 - 1.0)).toBeLessThan(1e-6);
            expect(Math.abs(row1 - 1.0)).toBeLessThan(1e-6);
        });
    });

    describe('activations', () => {
        it('relu should zero out negatives', () => {
            const t = new Tensor(new Float32Array([-2, -1, 0, 1, 2]), [5]);
            const r = Ops.relu(t);
            expect(Array.from(r.data)).toEqual([0, 0, 0, 1, 2]);
        });

        it('sigmoid should be 0.5 at x=0', () => {
            const t = new Tensor(new Float32Array([0]), [1]);
            const s = Ops.sigmoid(t);
            expect(Math.abs(s.data[0] - 0.5)).toBeLessThan(1e-6);
        });

        it('gelu(0) should be ~0', () => {
            const t = new Tensor(new Float32Array([0]), [1]);
            const g = Ops.gelu(t);
            expect(Math.abs(g.data[0])).toBeLessThan(1e-6);
        });
    });

    describe('layerNorm', () => {
        it('should normalize to mean~0 with identity scale/shift', () => {
            const x = Tensor.from([[1, 2, 3, 4]]);
            const gamma = Tensor.ones([4]);
            const beta = Tensor.zeros([4]);
            const result = Ops.layerNorm(x, gamma, beta);
            const mean = result.data.reduce((a, b) => a + b, 0) / 4;
            expect(Math.abs(mean)).toBeLessThan(1e-5);
        });
    });

    describe('chaining on Tensor.prototype', () => {
        it('should support method chaining', () => {
            const a = Tensor.from([
                [1, -1],
                [2, -2],
            ]);
            const result = a.relu();
            expect(result.toArray()).toEqual([
                [1, 0],
                [2, 0],
            ]);
        });

        it('should support dot chaining', () => {
            const a = Tensor.from([
                [1, 2],
                [3, 4],
            ]);
            const b = Tensor.from([
                [5, 6],
                [7, 8],
            ]);
            const c = a.dot(b);
            expect(c.shape).toEqual([2, 2]);
        });
    });
});
