/**
 * @file src/pipeline/Executor.js
 * @description Pipeline execution engine. Runs steps sequentially, passing results through context.
 */

import { PipelineError } from '../errors.js';
import { tokenizerHandler } from './handlers/TokenizerHandler.js';
import { logicGateHandler } from './handlers/LogicGateHandler.js';
import { neuralHandler } from './handlers/NeuralHandler.js';
import { vectorBuilderHandler } from './handlers/VectorBuilderHandler.js';
import { decoderHandler } from './handlers/DecoderHandler.js';
import { generatorHandler } from './handlers/GeneratorHandler.js';

const HandlerRegistry = new Map([
    ['tokenizer', tokenizerHandler],
    ['logic_gate', logicGateHandler],
    ['neural_model', neuralHandler],
    ['vector_builder', vectorBuilderHandler],
    ['decoder', decoderHandler],
    ['generator', generatorHandler],
]);

/**
 * Execute a processing pipeline.
 * @param {object} pipelineConfig - { input_field, pipeline: [...steps] }
 * @param {any} rawInput - Raw input data.
 * @param {object} [options]
 * @param {function} [options.onStepStart] - Called before each step: ({ step, index, total }).
 * @param {function} [options.onStepComplete] - Called after each step: ({ step, index, total, result }).
 * @param {function} [options.onProgress] - Called with progress fraction (0 to 1).
 * @returns {Promise<object>} Context object with all intermediate and final results.
 * @throws {PipelineError} On invalid config or step failure.
 */
export async function executePipeline(pipelineConfig, rawInput, options = {}) {
    if (!pipelineConfig || !Array.isArray(pipelineConfig.pipeline)) {
        throw new PipelineError("Invalid pipeline config: missing 'pipeline' array.");
    }

    const { onStepStart, onStepComplete, onProgress } = options;
    const totalSteps = pipelineConfig.pipeline.length;
    const context = {};
    context[pipelineConfig.input_field] = rawInput;

    for (let i = 0; i < totalSteps; i++) {
        const step = pipelineConfig.pipeline[i];

        if (!step.id || !step.type) {
            throw new PipelineError(
                `Pipeline step at index ${i} must have 'id' and 'type' fields.`,
                { stepId: step.id, stepType: step.type }
            );
        }

        const handler = HandlerRegistry.get(step.type);
        if (!handler) {
            throw new PipelineError(
                `No handler found for type "${step.type}". Available: ${[...HandlerRegistry.keys()].join(', ')}.`,
                { stepId: step.id, stepType: step.type }
            );
        }

        if (onStepStart) onStepStart({ step, index: i, total: totalSteps });

        try {
            const stepResult = await handler.process(step, context);
            context[step.id] = stepResult;
        } catch (err) {
            if (err instanceof PipelineError) throw err;
            throw new PipelineError(
                `Pipeline failed at step '${step.id}' (${step.type}): ${err.message}`,
                { stepId: step.id, stepType: step.type, cause: err }
            );
        }

        if (onStepComplete)
            onStepComplete({ step, index: i, total: totalSteps, result: context[step.id] });
        if (onProgress) onProgress((i + 1) / totalSteps);
    }

    return context;
}
