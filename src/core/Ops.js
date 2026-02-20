/**
 * @file src/core/Ops.js
 * @description Pure math operations for tensors. Forward-only, no autograd.
 */

import { Tensor } from './Tensor.js';
import { ShapeError } from '../errors.js';

const Ops = {

    /**
     * Matrix multiplication of two 2D tensors.
     * @param {Tensor} a - Shape [M, K]
     * @param {Tensor} b - Shape [K, N]
     * @returns {Tensor} Shape [M, N]
     * @throws {ShapeError} If tensors are not 2D or inner dimensions don't match.
     */
    matMul(a, b) {
        if (a.shape.length !== 2 || b.shape.length !== 2) {
            throw new ShapeError('matMul requires two 2D tensors.', {
                expected: '[M,K] x [K,N]',
                actual: `[${a.shape}] x [${b.shape}]`,
                operation: 'matMul',
            });
        }
        if (a.shape[1] !== b.shape[0]) {
            throw new ShapeError(
                `Inner dimensions must match for matMul: ${a.shape[1]} !== ${b.shape[0]}.`,
                { expected: `[${a.shape[0]},K] x [K,${b.shape[1]}]`, actual: `[${a.shape}] x [${b.shape}]`, operation: 'matMul' }
            );
        }

        const [M, K] = a.shape;
        const N = b.shape[1];
        const result = new Float32Array(M * N);

        for (let i = 0; i < M; i++) {
            for (let j = 0; j < N; j++) {
                let sum = 0;
                for (let k = 0; k < K; k++) {
                    sum += a.data[i * K + k] * b.data[k * N + j];
                }
                result[i * N + j] = sum;
            }
        }
        return new Tensor(result, [M, N]);
    },

    /**
     * Element-wise addition with broadcasting support.
     * Supports: [M,N]+[M,N], [M,N]+[1,N], [M,N]+[N].
     * @param {Tensor} a
     * @param {Tensor} b
     * @returns {Tensor}
     */
    add(a, b) {
        // Same shape
        if (a.size === b.size && a.shape.length === b.shape.length &&
            a.shape.every((d, i) => d === b.shape[i])) {
            const result = new Float32Array(a.size);
            for (let i = 0; i < a.size; i++) {
                result[i] = a.data[i] + b.data[i];
            }
            return new Tensor(result, a.shape);
        }

        // Broadcasting: [M, N] + [1, N] or [M, N] + [N]
        if (a.shape.length === 2) {
            const bCols = b.shape[b.shape.length - 1];
            if (a.shape[1] === bCols && (b.size === bCols)) {
                const [rows, cols] = a.shape;
                const result = new Float32Array(a.size);
                for (let i = 0; i < rows; i++) {
                    for (let j = 0; j < cols; j++) {
                        result[i * cols + j] = a.data[i * cols + j] + b.data[j];
                    }
                }
                return new Tensor(result, a.shape);
            }
        }

        throw new ShapeError('Incompatible shapes for add.', {
            expected: '[M,N]+[M,N] or [M,N]+[N]',
            actual: `[${a.shape}] + [${b.shape}]`,
            operation: 'add',
        });
    },

    /**
     * Element-wise multiplication. Supports scalar broadcasting.
     * @param {Tensor} a
     * @param {Tensor} b
     * @returns {Tensor}
     */
    mul(a, b) {
        // Same shape
        if (a.size === b.size && a.shape.length === b.shape.length &&
            a.shape.every((d, i) => d === b.shape[i])) {
            const result = new Float32Array(a.size);
            for (let i = 0; i < a.size; i++) {
                result[i] = a.data[i] * b.data[i];
            }
            return new Tensor(result, a.shape);
        }

        // Scalar broadcast
        if (b.size === 1) {
            const s = b.data[0];
            const result = new Float32Array(a.size);
            for (let i = 0; i < a.size; i++) result[i] = a.data[i] * s;
            return new Tensor(result, a.shape);
        }
        if (a.size === 1) {
            const s = a.data[0];
            const result = new Float32Array(b.size);
            for (let i = 0; i < b.size; i++) result[i] = b.data[i] * s;
            return new Tensor(result, b.shape);
        }

        throw new ShapeError('Incompatible shapes for mul.', {
            expected: 'same shape or scalar broadcast',
            actual: `[${a.shape}] * [${b.shape}]`,
            operation: 'mul',
        });
    },

    /**
     * Transpose a 2D tensor.
     * @param {Tensor} a - Shape [M, N]
     * @returns {Tensor} Shape [N, M]
     */
    transpose(a) {
        if (a.shape.length !== 2) {
            throw new ShapeError('transpose requires a 2D tensor.', {
                expected: '[M, N]', actual: `[${a.shape}]`, operation: 'transpose',
            });
        }
        const [rows, cols] = a.shape;
        const result = new Float32Array(rows * cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j * rows + i] = a.data[i * cols + j];
            }
        }
        return new Tensor(result, [cols, rows]);
    },

    /**
     * ReLU activation: max(0, x).
     * @param {Tensor} a
     * @returns {Tensor}
     */
    relu(a) {
        const result = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++) {
            result[i] = a.data[i] > 0 ? a.data[i] : 0;
        }
        return new Tensor(result, a.shape);
    },

    /**
     * GELU activation (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
     * @param {Tensor} a
     * @returns {Tensor}
     */
    gelu(a) {
        const result = new Float32Array(a.size);
        const c = Math.sqrt(2 / Math.PI);
        for (let i = 0; i < a.size; i++) {
            const x = a.data[i];
            result[i] = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
        }
        return new Tensor(result, a.shape);
    },

    /**
     * Sigmoid activation: 1 / (1 + exp(-x)).
     * @param {Tensor} a
     * @returns {Tensor}
     */
    sigmoid(a) {
        const result = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++) {
            result[i] = 1 / (1 + Math.exp(-a.data[i]));
        }
        return new Tensor(result, a.shape);
    },

    /**
     * Row-wise softmax for 1D or 2D tensors.
     * @param {Tensor} a
     * @returns {Tensor}
     */
    softmax(a) {
        const result = new Float32Array(a.size);

        if (a.shape.length === 2) {
            const [rows, cols] = a.shape;
            for (let i = 0; i < rows; i++) {
                const offset = i * cols;
                let maxVal = -Infinity;
                for (let j = 0; j < cols; j++) {
                    if (a.data[offset + j] > maxVal) maxVal = a.data[offset + j];
                }
                let sumExp = 0;
                for (let j = 0; j < cols; j++) {
                    const e = Math.exp(a.data[offset + j] - maxVal);
                    result[offset + j] = e;
                    sumExp += e;
                }
                for (let j = 0; j < cols; j++) {
                    result[offset + j] /= sumExp;
                }
            }
        } else if (a.shape.length === 1) {
            let maxVal = -Infinity;
            for (let i = 0; i < a.size; i++) {
                if (a.data[i] > maxVal) maxVal = a.data[i];
            }
            let sumExp = 0;
            for (let i = 0; i < a.size; i++) {
                const e = Math.exp(a.data[i] - maxVal);
                result[i] = e;
                sumExp += e;
            }
            for (let i = 0; i < a.size; i++) {
                result[i] /= sumExp;
            }
        } else {
            throw new ShapeError('softmax supports only 1D and 2D tensors.', {
                actual: `[${a.shape}]`, operation: 'softmax',
            });
        }

        return new Tensor(result, a.shape);
    },

    /**
     * Layer normalization over the last dimension of a 2D tensor.
     * @param {Tensor} x - Shape [rows, features]
     * @param {Tensor} gamma - Shape [features] or [1, features]
     * @param {Tensor} beta - Shape [features] or [1, features]
     * @param {number} [eps=1e-5]
     * @returns {Tensor}
     */
    layerNorm(x, gamma, beta, eps = 1e-5) {
        const [rows, cols] = x.shape;
        const result = new Float32Array(x.size);

        for (let i = 0; i < rows; i++) {
            const offset = i * cols;

            // Mean
            let sum = 0;
            for (let j = 0; j < cols; j++) sum += x.data[offset + j];
            const mean = sum / cols;

            // Variance
            let varSum = 0;
            for (let j = 0; j < cols; j++) {
                const diff = x.data[offset + j] - mean;
                varSum += diff * diff;
            }
            const stdInv = 1 / Math.sqrt(varSum / cols + eps);

            // Normalize, scale, shift
            for (let j = 0; j < cols; j++) {
                const normalized = (x.data[offset + j] - mean) * stdInv;
                result[offset + j] = normalized * gamma.data[j] + beta.data[j];
            }
        }

        return new Tensor(result, x.shape);
    }
};

// Convenience methods on Tensor.prototype
Tensor.prototype.dot = function (other) { return Ops.matMul(this, other); };
Tensor.prototype.add = function (other) { return Ops.add(this, other); };
Tensor.prototype.mul = function (other) { return Ops.mul(this, other); };
Tensor.prototype.transpose = function () { return Ops.transpose(this); };
Tensor.prototype.relu = function () { return Ops.relu(this); };
Tensor.prototype.gelu = function () { return Ops.gelu(this); };
Tensor.prototype.sigmoid = function () { return Ops.sigmoid(this); };
Tensor.prototype.softmax = function () { return Ops.softmax(this); };

export { Ops };
