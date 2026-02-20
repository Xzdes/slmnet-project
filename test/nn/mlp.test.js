import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/core/Tensor.js';
import { Ops } from '../../src/core/Ops.js';
import { MLP } from '../../src/nn/MLP.js';

describe('MLP', () => {
    describe('fromLegacy', () => {
        it('should build from legacy config and run forward pass', () => {
            const config = {
                architecture: { input: 3, hidden: 2, output: 1 },
                weights: {
                    ih: [1, 0, 0, 0, 1, 0], // [hidden=2, input=3]
                    ho: [1, 1],              // [output=1, hidden=2]
                    bh: [0, 0],
                    bo: [0],
                },
            };

            const mlp = MLP.fromLegacy(config);
            const x = new Tensor(new Float32Array([1, 1, 0]), [1, 3]);
            const out = mlp.forward(x);
            expect(out.shape).toEqual([1, 1]);
            // After sigmoid activations, output should be in (0, 1)
            expect(out.data[0]).toBeGreaterThan(0);
            expect(out.data[0]).toBeLessThan(1);
        });
    });

    describe('forward', () => {
        it('should pass through multiple layers with activations', () => {
            const { Linear } = require('../../src/nn/Linear.js');

            const l1 = new Linear(new Float32Array([1, 0, 0, 1]), null, 2, 2);
            const l2 = new Linear(new Float32Array([1, 1]), null, 2, 1);

            const mlp = new MLP([
                { linear: l1, activation: 'relu' },
                { linear: l2, activation: 'sigmoid' },
            ]);

            const x = new Tensor(new Float32Array([3, -1]), [1, 2]);
            const out = mlp.forward(x);
            expect(out.shape).toEqual([1, 1]);
            // ReLU([3, -1]) = [3, 0], then matmul with [1,1] = 3, sigmoid(3) ≈ 0.95
            expect(out.data[0]).toBeGreaterThan(0.9);
        });
    });
});
