import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';

describe('Tensor', () => {
    describe('constructor', () => {
        it('should create from flat data and shape', () => {
            const t = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            expect(t.shape).toEqual([2, 3]);
            expect(t.size).toBe(6);
        });

        it('should throw on shape/data mismatch', () => {
            expect(() => new Tensor(new Float32Array([1, 2, 3]), [2, 3])).toThrow('Data size');
        });

        it('should infer shape from nested array', () => {
            const t = new Tensor([[1, 2], [3, 4], [5, 6]]);
            expect(t.shape).toEqual([3, 2]);
            expect(t.size).toBe(6);
        });

        it('should create 1D tensor', () => {
            const t = new Tensor(new Float32Array([1, 2, 3]), [3]);
            expect(t.shape).toEqual([3]);
            expect(t.size).toBe(3);
        });

        it('should convert regular array to Float32Array', () => {
            const t = new Tensor([1, 2, 3, 4], [2, 2]);
            expect(t.data).toBeInstanceOf(Float32Array);
        });
    });

    describe('static factories', () => {
        it('zeros creates zero-filled tensor', () => {
            const t = Tensor.zeros([3, 4]);
            expect(t.shape).toEqual([3, 4]);
            expect(t.data.every(v => v === 0)).toBe(true);
        });

        it('ones creates ones-filled tensor', () => {
            const t = Tensor.ones([2, 2]);
            expect(t.data.every(v => v === 1)).toBe(true);
        });

        it('from creates from nested array', () => {
            const t = Tensor.from([[1, 2], [3, 4]]);
            expect(t.shape).toEqual([2, 2]);
            expect(Array.from(t.data)).toEqual([1, 2, 3, 4]);
        });
    });

    describe('reshape', () => {
        it('should reshape without copying data', () => {
            const t = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
            const r = t.reshape([3, 2]);
            expect(r.shape).toEqual([3, 2]);
            expect(r.data).toBe(t.data);
        });

        it('should throw on incompatible reshape', () => {
            const t = Tensor.zeros([2, 3]);
            expect(() => t.reshape([2, 4])).toThrow('Cannot reshape');
        });
    });

    describe('toArray', () => {
        it('should convert 2D tensor to nested array', () => {
            const t = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
            expect(t.toArray()).toEqual([[1, 2], [3, 4]]);
        });

        it('should convert 1D tensor to flat array', () => {
            const t = new Tensor(new Float32Array([1, 2, 3]), [3]);
            expect(t.toArray()).toEqual([1, 2, 3]);
        });

        it('should handle 3D tensors', () => {
            const t = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
            expect(t.toArray()).toEqual([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        });
    });
});
