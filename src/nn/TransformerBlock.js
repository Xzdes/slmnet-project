/**
 * @file src/nn/TransformerBlock.js
 * @description Pre-LN Transformer Block. Forward-only.
 * Architecture: x + Attention(LN(x)), then x + FFN(LN(x))
 */

import { MultiHeadAttention } from './Attention.js';
import { FeedForward } from './FeedForward.js';
import { LayerNorm } from './LayerNorm.js';

class TransformerBlock {
    /**
     * @param {MultiHeadAttention} attention
     * @param {FeedForward} ffn
     * @param {LayerNorm} ln1
     * @param {LayerNorm} ln2
     */
    constructor(attention, ffn, ln1, ln2) {
        this.attention = attention;
        this.ffn = ffn;
        this.ln1 = ln1;
        this.ln2 = ln2;
    }

    /**
     * @param {Tensor} x - Shape [seqLen, embedDim]
     * @returns {Tensor} Shape [seqLen, embedDim]
     */
    forward(x) {
        // Pre-LN: normalize before sublayer
        const norm1 = this.ln1.forward(x);
        const attnOut = this.attention.forward(norm1);
        const x1 = x.add(attnOut); // Residual connection

        const norm2 = this.ln2.forward(x1);
        const ffnOut = this.ffn.forward(norm2);
        const x2 = x1.add(ffnOut); // Residual connection

        return x2;
    }
}

export { TransformerBlock };
