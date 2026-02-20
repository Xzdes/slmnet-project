/**
 * @file src/nn/MLP.js
 * @description Simple multi-layer perceptron. Forward-only.
 * Compatible with slmnet-project legacy model format.
 */

import { Ops } from '../core/Ops.js';
import { Linear } from './Linear.js';

const ACTIVATIONS = {
    relu: Ops.relu,
    gelu: Ops.gelu,
    sigmoid: Ops.sigmoid,
    none: (x) => x,
};

class MLP {
    /**
     * @param {Array<{linear: Linear, activation: string}>} layers
     */
    constructor(layers) {
        this.layers = layers;
    }

    /**
     * Build an MLP from the legacy JSON format (slmnet-project compatibility).
     * @param {object} config - { architecture: {input, hidden, output}, weights: {ih, ho, bh, bo} }
     * @returns {MLP}
     */
    static fromLegacy(config) {
        const { architecture: arch, weights } = config;

        // Legacy format stores weights as [hidden, input] — needs transpose.
        // We store as [input, hidden], so we transpose during construction.
        const ihData = new Float32Array(weights.ih);
        const hoData = new Float32Array(weights.ho);

        // Transpose ih: [hidden, input] → [input, hidden]
        const ihTransposed = new Float32Array(arch.input * arch.hidden);
        for (let i = 0; i < arch.hidden; i++) {
            for (let j = 0; j < arch.input; j++) {
                ihTransposed[j * arch.hidden + i] = ihData[i * arch.input + j];
            }
        }

        // Transpose ho: [output, hidden] → [hidden, output]
        const hoTransposed = new Float32Array(arch.hidden * arch.output);
        for (let i = 0; i < arch.output; i++) {
            for (let j = 0; j < arch.hidden; j++) {
                hoTransposed[j * arch.output + i] = hoData[i * arch.hidden + j];
            }
        }

        const layers = [
            {
                linear: new Linear(
                    ihTransposed,
                    new Float32Array(weights.bh),
                    arch.input,
                    arch.hidden
                ),
                activation: 'sigmoid',
            },
            {
                linear: new Linear(
                    hoTransposed,
                    new Float32Array(weights.bo),
                    arch.hidden,
                    arch.output
                ),
                activation: 'sigmoid',
            },
        ];

        return new MLP(layers);
    }

    /**
     * @param {Tensor} x - Shape [batch, inputDim]
     * @returns {Tensor}
     */
    forward(x) {
        let out = x;
        for (const layer of this.layers) {
            out = layer.linear.forward(out);
            const activationFn = ACTIVATIONS[layer.activation] || ACTIVATIONS.none;
            out = activationFn(out);
        }
        return out;
    }
}

export { MLP };
