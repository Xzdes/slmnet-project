/**
 * @file src/pipeline/handlers/TokenizerHandler.js
 * @description Pipeline handler for text-to-vector conversion (Bag of Words).
 */

import { Loader } from '../../Loader.js';
import { PipelineError } from '../../errors.js';
import { getValueFromContext } from '../ContextNavigator.js';

const vocabCache = new Map();

/**
 * Load vocabulary and build a Map for O(1) lookups.
 * @param {string} vocabUrl
 * @returns {Promise<{ vocab: string[], vocabMap: Map<string, number> }>}
 */
async function getVocabulary(vocabUrl) {
    if (vocabCache.has(vocabUrl)) return vocabCache.get(vocabUrl);
    const vocabData = await Loader.loadJson(vocabUrl);
    if (!Array.isArray(vocabData)) {
        throw new PipelineError(`Vocabulary at ${vocabUrl} must be an array.`);
    }
    const vocabMap = new Map();
    vocabData.forEach((word, idx) => vocabMap.set(word, idx));
    const entry = { vocab: vocabData, vocabMap };
    vocabCache.set(vocabUrl, entry);
    return entry;
}

/**
 * Convert text to a Bag-of-Words binary vector using O(1) Map lookup.
 * @param {string} text
 * @param {Map<string, number>} vocabMap
 * @param {number} vectorSize
 * @returns {number[]}
 */
function textToBoW(text, vocabMap, vectorSize) {
    const vector = new Array(vectorSize).fill(0);
    const lower = text.toLowerCase().replace(/[.,!?"'()`]/g, '');
    const words = new Set(lower.split(/\s+/));
    for (const word of words) {
        const index = vocabMap.get(word);
        if (index !== undefined && index < vectorSize) {
            vector[index] = 1;
        }
    }
    return vector;
}

class TokenizerHandler {
    async process(step, context) {
        if (!step.vocab_url || !step.output_size) {
            throw new PipelineError(`TokenizerHandler requires 'vocab_url' and 'output_size'.`, {
                stepId: step.id,
                stepType: step.type,
            });
        }
        const inputText = getValueFromContext(context, step.input);
        if (typeof inputText !== 'string') {
            throw new PipelineError(
                `TokenizerHandler: input must be a string, got ${typeof inputText}.`,
                { stepId: step.id, stepType: step.type }
            );
        }
        const { vocabMap } = await getVocabulary(step.vocab_url);
        return textToBoW(inputText, vocabMap, step.output_size);
    }
}

export const tokenizerHandler = new TokenizerHandler();
