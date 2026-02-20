/**
 * @file src/core/Tensor.js
 * @description Forward-only N-dimensional data container.
 * No autograd, no gradients — optimized for inference.
 */

class Tensor {
    /**
     * @param {Float32Array|Array|number[]} data - Flat data array.
     * @param {number[]} shape - Tensor shape.
     */
    constructor(data, shape) {
        if (shape) {
            const expectedSize = shape.reduce((a, b) => a * b, 1);
            if (data.length !== expectedSize) {
                throw new Error(
                    `Data size (${data.length}) does not match shape [${shape}] (expected ${expectedSize}).`
                );
            }
            this.data = data instanceof Float32Array ? data : new Float32Array(data);
            this.shape = shape;
        } else {
            const { flatData, inferredShape } = Tensor._inferShapeAndFlatten(data);
            this.data = new Float32Array(flatData);
            this.shape = inferredShape;
        }
        this.size = this.data.length;
    }

    /**
     * Create a tensor from a nested array.
     * @param {Array} arr
     * @returns {Tensor}
     */
    static from(arr) {
        return new Tensor(arr);
    }

    /**
     * Create a zero-filled tensor.
     * @param {number[]} shape
     * @returns {Tensor}
     */
    static zeros(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float32Array(size), shape);
    }

    /**
     * Create a ones-filled tensor.
     * @param {number[]} shape
     * @returns {Tensor}
     */
    static ones(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float32Array(size).fill(1), shape);
    }

    /**
     * Reshape tensor without copying data.
     * @param {number[]} newShape
     * @returns {Tensor}
     */
    reshape(newShape) {
        const newSize = newShape.reduce((a, b) => a * b, 1);
        if (this.size !== newSize) {
            throw new Error(
                `Cannot reshape [${this.shape}] (size ${this.size}) to [${newShape}] (size ${newSize}).`
            );
        }
        return new Tensor(this.data, newShape);
    }

    /**
     * Return data as a nested JS array.
     * @returns {Array}
     */
    toArray() {
        const build = (shape, offset) => {
            if (shape.length === 1) {
                return Array.from(this.data.slice(offset, offset + shape[0]));
            }
            const result = [];
            const stride = shape.slice(1).reduce((a, b) => a * b, 1);
            for (let i = 0; i < shape[0]; i++) {
                result.push(build(shape.slice(1), offset + i * stride));
            }
            return result;
        };
        return build(this.shape, 0);
    }

    /* eslint-disable no-console */
    print() {
        console.log('Tensor {');
        console.log('  shape:', this.shape);
        console.log('  data:', this.toArray());
        console.log('}');
    }
    /* eslint-enable no-console */

    /** @private */
    static _inferShapeAndFlatten(arr) {
        const flatData = [];
        const inferredShape = [];
        let currentLevel = arr;
        while (Array.isArray(currentLevel)) {
            if (currentLevel.length === 0) break;
            inferredShape.push(currentLevel.length);
            if (Array.isArray(currentLevel[0])) {
                const firstLen = currentLevel[0].length;
                for (let i = 1; i < currentLevel.length; i++) {
                    if (!Array.isArray(currentLevel[i]) || currentLevel[i].length !== firstLen) {
                        throw new Error(
                            'Nested arrays have different lengths. Cannot create tensor.'
                        );
                    }
                }
            }
            currentLevel = currentLevel[0];
        }
        const flatten = (sub) => {
            for (const el of sub) {
                if (Array.isArray(el)) flatten(el);
                else flatData.push(el);
            }
        };
        flatten(arr);
        return { flatData, inferredShape };
    }
}

export { Tensor };
