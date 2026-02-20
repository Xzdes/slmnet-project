/**
 * @file src/nn/FeedForward.js
 * @description Position-wise Feed-Forward Network. Forward-only.
 * Architecture: Linear → GELU → Linear
 */

import { Linear } from './Linear.js';

class FeedForward {
    /**
     * @param {Linear} linear1 - [embedDim, hiddenDim]
     * @param {Linear} linear2 - [hiddenDim, embedDim]
     */
    constructor(linear1, linear2) {
        this.linear1 = linear1;
        this.linear2 = linear2;
    }

    /**
     * @param {Tensor} x - Shape [seqLen, embedDim]
     * @returns {Tensor} Shape [seqLen, embedDim]
     */
    forward(x) {
        let out = this.linear1.forward(x);
        out = out.gelu();
        out = this.linear2.forward(out);
        return out;
    }
}

export { FeedForward };
