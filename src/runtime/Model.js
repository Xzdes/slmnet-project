/**
 * @file src/runtime/Model.js
 * @description High-level Model class. Load a .slmnet file, run classification or generation.
 */

import { Tensor } from '../core/Tensor.js';
import { Ops } from '../core/Ops.js';
import { ModelFormat, ARCH_TYPE, TOKENIZER_TYPE } from './ModelFormat.js';
import { Sampler } from './Sampler.js';
import { Linear } from '../nn/Linear.js';
import { Embedding } from '../nn/Embedding.js';
import { LayerNorm } from '../nn/LayerNorm.js';
import { MultiHeadAttention } from '../nn/Attention.js';
import { FeedForward } from '../nn/FeedForward.js';
import { TransformerBlock } from '../nn/TransformerBlock.js';
import { MLP } from '../nn/MLP.js';
import { CharTokenizer } from '../tokenizer/CharTokenizer.js';
import { BPETokenizer } from '../tokenizer/BPETokenizer.js';
import { ModelLoadError, ValidationError } from '../errors.js';

class Model {
    constructor() {
        this.header = null;
        this.tokenizer = null;
        this.labels = null;
        this.network = null; // { type, forward(x) }
    }

    /**
     * Load a model from a URL (.slmnet binary) or a JSON config (legacy).
     * @param {string} url - URL to .slmnet or .json file.
     * @param {object} [options]
     * @param {function} [options.onProgress] - Progress callback (0 to 1).
     * @returns {Promise<Model>}
     * @throws {ModelLoadError} On network or parse failure.
     */
    static async load(url, { onProgress } = {}) {
        if (typeof url !== 'string' || !url.trim()) {
            throw new ModelLoadError('Model URL must be a non-empty string.', { url });
        }

        if (onProgress) onProgress(0);

        const response = await fetch(url);
        if (!response.ok) {
            throw new ModelLoadError(
                `Failed to load model: ${response.statusText}`,
                { url, status: response.status }
            );
        }

        const model = new Model();

        if (url.endsWith('.json')) {
            const config = await response.json();
            if (onProgress) onProgress(0.5);
            model._loadLegacyJSON(config);
        } else {
            const buffer = await response.arrayBuffer();
            if (onProgress) onProgress(0.5);
            model._loadBinary(buffer);
        }

        if (onProgress) onProgress(1);
        return model;
    }

    /**
     * Load model from an ArrayBuffer (.slmnet binary).
     * @param {ArrayBuffer} buffer
     * @returns {Model}
     */
    static fromBuffer(buffer) {
        const model = new Model();
        model._loadBinary(buffer);
        return model;
    }

    /**
     * Classify text and return label + scores.
     * @param {string} text
     * @returns {{ label: string, score: number, scores: object }}
     */
    classify(text) {
        if (!this.labels || this.labels.length === 0) {
            throw new ValidationError('This model does not have classification labels.');
        }

        const inputTensor = this._textToTensor(text);
        const output = this.network.forward(inputTensor);

        // Apply softmax to get probabilities
        const probs = output.softmax();

        // Find best label
        let maxIdx = 0;
        for (let i = 1; i < probs.size; i++) {
            if (probs.data[i] > probs.data[maxIdx]) maxIdx = i;
        }

        const scores = {};
        for (let i = 0; i < this.labels.length && i < probs.size; i++) {
            scores[this.labels[i]] = probs.data[i];
        }

        return {
            label: this.labels[maxIdx],
            score: probs.data[maxIdx],
            scores
        };
    }

    /**
     * Generate text from a prompt.
     * @param {string} prompt
     * @param {object} [options]
     * @param {number} [options.maxTokens=100]
     * @param {number} [options.temperature=0.8]
     * @param {number} [options.topK=10]
     * @param {number} [options.topP=1.0]
     * @param {function} [options.onToken] - Called with each new token string.
     * @param {function} [options.shouldStop] - Return true to stop generation early.
     * @returns {string}
     * @throws {ValidationError} If no tokenizer or empty prompt.
     */
    generate(prompt, { maxTokens = 100, temperature = 0.8, topK = 10, topP = 1.0, onToken, shouldStop } = {}) {
        if (!this.tokenizer || !this.tokenizer.encode) {
            throw new ValidationError('This model does not have a tokenizer for generation.');
        }

        const blockSize = this.header.blockSize || 64;
        const promptIds = this.tokenizer.encode(prompt);

        if (promptIds.length === 0) {
            throw new ValidationError(
                'Prompt encoded to zero tokens. Check that the prompt contains characters in the tokenizer vocabulary.',
                { prompt: prompt.slice(0, 100) }
            );
        }

        const contextIds = [...promptIds];

        for (let i = 0; i < maxTokens; i++) {
            // Take last blockSize tokens as context
            const currentContext = contextIds.slice(-blockSize);
            // 1D tensor [seqLen] for transformer
            const contextTensor = new Tensor(
                new Float32Array(currentContext),
                [currentContext.length]
            );

            // Forward pass -> [seqLen, vocabSize]
            const logits = this.network.forward(contextTensor);

            // Get logits for the last position
            const vocabSize = this.header.vocabSize;
            const lastLogits = new Tensor(
                logits.data.slice((logits.shape[0] - 1) * vocabSize, logits.shape[0] * vocabSize),
                [vocabSize]
            );

            // Sample next token
            const nextId = Sampler.sample(lastLogits, { temperature, topK, topP });
            contextIds.push(nextId);

            if (onToken) onToken(this.tokenizer.decode([nextId]));
            if (shouldStop) {
                const generatedIds = contextIds.slice(promptIds.length);
                if (shouldStop(this.tokenizer.decode(generatedIds))) break;
            }
        }

        const generatedIds = contextIds.slice(promptIds.length);
        return prompt + this.tokenizer.decode(generatedIds);
    }

    /**
     * Low-level forward pass.
     * @param {Tensor} input
     * @returns {Tensor}
     */
    forward(input) {
        return this.network.forward(input);
    }

    // --- Private Methods ---

    /** @private */
    _loadBinary(buffer) {
        const parsed = ModelFormat.parse(buffer);
        this.header = parsed.header;
        this.labels = parsed.labels;

        // Build tokenizer
        this._buildTokenizer(parsed.header.tokenizerType, parsed.tokenizerConfig);

        // Build network
        if (parsed.header.archType === ARCH_TYPE.TRANSFORMER) {
            this._buildTransformer(parsed.header, parsed.weights);
        } else {
            this._buildMLP(parsed.header, parsed.weights);
        }
    }

    /** @private */
    _loadLegacyJSON(config) {
        // Legacy slmnet-project format: { architecture: {input,hidden,output}, weights: {ih,ho,bh,bo} }
        this.header = {
            archType: ARCH_TYPE.MLP,
            vocabSize: config.architecture.input,
            numLabels: config.architecture.output,
        };
        this.labels = config.labels || null;

        const mlp = MLP.fromLegacy(config);
        this.network = { forward: (x) => mlp.forward(x) };
    }

    /** @private */
    _buildTokenizer(type, config) {
        switch (type) {
            case TOKENIZER_TYPE.CHAR:
                this.tokenizer = CharTokenizer.fromConfig(config);
                break;
            case TOKENIZER_TYPE.BPE:
                this.tokenizer = BPETokenizer.fromConfig(config);
                break;
            case TOKENIZER_TYPE.BOW: {
                // BoW tokenizer: Map-based O(1) lookup for Bag-of-Words conversion
                const vocabMap = new Map();
                config.vocab.forEach((word, idx) => vocabMap.set(word, idx));
                this.tokenizer = {
                    vocab: config.vocab,
                    vocabSize: config.vocab.length,
                    encode: (text) => {
                        const vectorSize = config.vectorSize || config.vocab.length;
                        const vector = new Array(vectorSize).fill(0);
                        const lower = text.toLowerCase().replace(/[.,!?"'()`]/g, '');
                        const words = new Set(lower.split(/\s+/));
                        for (const word of words) {
                            const idx = vocabMap.get(word);
                            if (idx !== undefined && idx < vectorSize) vector[idx] = 1;
                        }
                        return vector;
                    },
                    decode: (ids) => ids.join(''),
                };
                break;
            }
            default:
                this.tokenizer = null;
        }
    }

    /** @private */
    _buildTransformer(header, weights) {
        const { vocabSize, embedDim, numHeads, numLayers, blockSize } = header;
        const headDim = embedDim / numHeads;

        const tokenEmbed = new Embedding(weights.get('token_embed').data, vocabSize, embedDim);
        const posEmbed = new Embedding(weights.get('pos_embed').data, blockSize, embedDim);

        const blocks = [];
        for (let i = 0; i < numLayers; i++) {
            const prefix = `block_${i}`;

            const wq = this._makeLinear(weights, `${prefix}.attn.wq`, embedDim, embedDim);
            const wk = this._makeLinear(weights, `${prefix}.attn.wk`, embedDim, embedDim);
            const wv = this._makeLinear(weights, `${prefix}.attn.wv`, embedDim, embedDim);
            const wo = this._makeLinear(weights, `${prefix}.attn.wo`, embedDim, embedDim);
            const attn = new MultiHeadAttention(wq, wk, wv, wo, numHeads, headDim);

            const hiddenDim = header.hiddenDim || embedDim * 4;
            const ff1 = this._makeLinear(weights, `${prefix}.ffn.linear1`, embedDim, hiddenDim);
            const ff2 = this._makeLinear(weights, `${prefix}.ffn.linear2`, hiddenDim, embedDim);
            const ffn = new FeedForward(ff1, ff2);

            const ln1 = this._makeLayerNorm(weights, `${prefix}.ln1`, embedDim);
            const ln2 = this._makeLayerNorm(weights, `${prefix}.ln2`, embedDim);

            blocks.push(new TransformerBlock(attn, ffn, ln1, ln2));
        }

        const finalLN = this._makeLayerNorm(weights, 'final_ln', embedDim);
        const outputHead = this._makeLinear(weights, 'output_head', embedDim, vocabSize);

        this.network = {
            forward: (inputTensor) => {
                // inputTensor: [seqLen] of token IDs (1D)
                const seqLen = inputTensor.shape[0];

                // Token embeddings: [seqLen] -> [seqLen, embedDim]
                let x = tokenEmbed.forward(inputTensor);

                // Position embeddings: [seqLen] -> [seqLen, embedDim]
                const posIds = new Float32Array(seqLen);
                for (let i = 0; i < seqLen; i++) posIds[i] = i;
                const posT = new Tensor(posIds, [seqLen]);
                const posE = posEmbed.forward(posT);
                x = x.add(posE);

                // Transformer blocks: [seqLen, embedDim] -> [seqLen, embedDim]
                for (const block of blocks) {
                    x = block.forward(x);
                }

                // Final layer norm
                x = finalLN.forward(x);

                // Output projection: [seqLen, embedDim] -> [seqLen, vocabSize]
                x = outputHead.forward(x);

                return x;
            }
        };
    }

    /** @private */
    _buildMLP(header, weights) {
        const layers = [];
        const layerNames = [...weights.keys()]
            .filter(k => k.startsWith('layer_'))
            .map(k => k.replace(/\.(weights|bias)$/, ''))
            .filter((v, i, a) => a.indexOf(v) === i)
            .sort();

        for (const layerName of layerNames) {
            const w = weights.get(`${layerName}.weights`);
            const b = weights.get(`${layerName}.bias`);
            const inF = w.shape[0];
            const outF = w.shape[1];
            const linear = new Linear(w.data, b ? b.data : null, inF, outF);

            // Determine activation from name or default
            const activation = layerName.includes('output') ? 'sigmoid' : 'relu';
            layers.push({ linear, activation });
        }

        const mlp = new MLP(layers);
        this.network = { forward: (x) => mlp.forward(x) };
    }

    /** @private */
    _makeLinear(weights, name, inF, outF) {
        const w = weights.get(`${name}.weights`);
        const b = weights.has(`${name}.bias`) ? weights.get(`${name}.bias`) : null;
        return new Linear(w.data, b ? b.data : null, inF, outF);
    }

    /** @private */
    _makeLayerNorm(weights, name, dim) {
        const gamma = weights.get(`${name}.gamma`);
        const beta = weights.get(`${name}.beta`);
        return new LayerNorm(gamma.data, beta.data, dim);
    }

    /** @private */
    _textToTensor(text) {
        if (!this.tokenizer || !this.tokenizer.encode) {
            throw new ValidationError('No tokenizer available for text encoding.');
        }
        const encoded = this.tokenizer.encode(text);
        if (encoded.length === 0) {
            throw new ValidationError(
                'Text encoded to zero tokens. The input may not contain any known vocabulary terms.',
                { textPreview: text.slice(0, 100) }
            );
        }
        // Transformer expects 1D [seqLen], MLP expects 2D [1, vectorSize]
        if (this.header.archType === ARCH_TYPE.TRANSFORMER) {
            return new Tensor(new Float32Array(encoded), [encoded.length]);
        }
        return new Tensor(new Float32Array(encoded), [1, encoded.length]);
    }
}

export { Model };
