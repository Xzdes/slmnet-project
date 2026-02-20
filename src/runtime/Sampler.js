/**
 * @file src/runtime/Sampler.js
 * @description Sampling strategies for text generation: Top-K, Top-P, temperature.
 */

import { Ops } from '../core/Ops.js';
import { Tensor } from '../core/Tensor.js';

class Sampler {
    /**
     * Sample a token ID from logits using temperature, Top-K, and Top-P.
     * @param {Tensor} logits - Shape [vocabSize] or [1, vocabSize]
     * @param {object} options
     * @param {number} [options.temperature=1.0]
     * @param {number} [options.topK=0] - 0 means no filtering.
     * @param {number} [options.topP=1.0] - 1.0 means no filtering.
     * @returns {number} Sampled token ID.
     */
    static sample(logits, { temperature = 1.0, topK = 0, topP = 1.0 } = {}) {
        const vocabSize = logits.size;
        const data = new Float32Array(logits.data);

        // 1. Apply temperature
        if (temperature !== 1.0 && temperature > 0) {
            for (let i = 0; i < vocabSize; i++) {
                data[i] /= temperature;
            }
        }

        // 2. Build indexed array for sorting
        const indexed = new Array(vocabSize);
        for (let i = 0; i < vocabSize; i++) {
            indexed[i] = { idx: i, val: data[i] };
        }
        indexed.sort((a, b) => b.val - a.val);

        // 3. Top-K filtering
        let candidates = indexed;
        if (topK > 0 && topK < vocabSize) {
            candidates = indexed.slice(0, topK);
        }

        // 4. Compute softmax on candidates
        const maxVal = candidates[0].val;
        let sumExp = 0;
        for (const c of candidates) {
            c.prob = Math.exp(c.val - maxVal);
            sumExp += c.prob;
        }
        for (const c of candidates) {
            c.prob /= sumExp;
        }

        // 5. Top-P (nucleus) filtering
        if (topP < 1.0) {
            let cumProb = 0;
            let cutoff = candidates.length;
            for (let i = 0; i < candidates.length; i++) {
                cumProb += candidates[i].prob;
                if (cumProb >= topP) {
                    cutoff = i + 1;
                    break;
                }
            }
            candidates = candidates.slice(0, cutoff);

            // Re-normalize
            let newSum = 0;
            for (const c of candidates) newSum += c.prob;
            for (const c of candidates) c.prob /= newSum;
        }

        // 6. Categorical sampling
        const rand = Math.random();
        let cumulative = 0;
        for (const c of candidates) {
            cumulative += c.prob;
            if (rand < cumulative) return c.idx;
        }

        // Fallback to top candidate
        return candidates[0].idx;
    }
}

export { Sampler };
