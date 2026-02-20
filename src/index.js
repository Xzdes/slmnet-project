/**
 * @file src/index.js
 * @description slmnet — Lightweight JS inference runtime for neural networks.
 * Main entry point and public API.
 *
 * @example
 * // Via <script> tag:
 * // <script src="https://cdn.jsdelivr.net/npm/slmnet@1/dist/slmnet.min.js"></script>
 * const model = await slmnet.Model.load('/my-model.slmnet');
 * const result = model.classify('hello world');
 *
 * @example
 * // Via ES module:
 * import slmnet from 'slmnet';
 * const ctx = await slmnet.run('./pipeline.json', 'input text');
 */

import { Model } from './runtime/Model.js';
import { Tensor } from './core/Tensor.js';
import { Ops } from './core/Ops.js';
import { Sampler } from './runtime/Sampler.js';
import { ModelFormat, ARCH_TYPE, QUANT_TYPE, TOKENIZER_TYPE } from './runtime/ModelFormat.js';
import { Loader } from './Loader.js';
import { executePipeline } from './pipeline/Executor.js';
import { SlmnetError, ShapeError, ModelLoadError, ValidationError, PipelineError } from './errors.js';

const slmnet = {
    /** High-level model class: load, classify, generate. */
    Model,

    /** N-dimensional tensor container. */
    Tensor,

    /** Pure math operations on tensors. */
    Ops,

    /** Top-k/top-p/temperature sampling. */
    Sampler,

    /** Binary .slmnet format parse/build. */
    ModelFormat,
    ARCH_TYPE,
    QUANT_TYPE,
    TOKENIZER_TYPE,

    /** Resource loader with caching. */
    Loader,

    /** Error classes for instanceof checks. */
    SlmnetError,
    ShapeError,
    ModelLoadError,
    ValidationError,
    PipelineError,

    /**
     * Run a processing pipeline.
     * @param {object|string} pipelineSource - Pipeline config object or URL to JSON.
     * @param {any} input - Raw input data.
     * @param {object} [options] - Pipeline options.
     * @param {function} [options.onStepStart] - Called before each step.
     * @param {function} [options.onStepComplete] - Called after each step.
     * @param {function} [options.onProgress] - Called with progress 0..1.
     * @returns {Promise<object>} Context with all intermediate and final results.
     * @throws {ValidationError} If pipelineSource is invalid.
     * @throws {PipelineError} If a pipeline step fails.
     */
    async run(pipelineSource, input, options = {}) {
        let config;
        if (typeof pipelineSource === 'string') {
            config = await Loader.loadJson(pipelineSource);
        } else if (typeof pipelineSource === 'object' && pipelineSource !== null) {
            config = pipelineSource;
        } else {
            throw new ValidationError(
                'Pipeline source must be a URL string or a config object.',
                { received: typeof pipelineSource }
            );
        }
        return executePipeline(config, input, options);
    },

    version: '1.0.0',
};

export default slmnet;
export { Model, Tensor, Ops, Sampler, ModelFormat, Loader };
