/**
 * @file src/pipeline/handlers/NeuralHandler.js
 * @description Pipeline handler for neural network inference (MLP or Transformer).
 */

import { Loader } from '../../Loader.js';
import { Model } from '../../runtime/Model.js';
import { Tensor } from '../../core/Tensor.js';
import { MLP } from '../../nn/MLP.js';
import { PipelineError } from '../../errors.js';
import { getValueFromContext } from '../ContextNavigator.js';

const modelCache = new Map();

async function getModel(modelUrl) {
    if (modelCache.has(modelUrl)) return modelCache.get(modelUrl);

    let model;
    if (modelUrl.endsWith('.slmnet')) {
        model = await Model.load(modelUrl);
    } else {
        const config = await Loader.loadJson(modelUrl);
        model = { network: MLP.fromLegacy(config) };
    }

    modelCache.set(modelUrl, model);
    return model;
}

class NeuralHandler {
    async process(step, context) {
        if (!step.model_url || !step.input) {
            throw new PipelineError(
                "NeuralHandler requires 'model_url' and 'input'.",
                { stepId: step.id, stepType: step.type }
            );
        }

        const model = await getModel(step.model_url);
        const inputVector = getValueFromContext(context, step.input);

        if (!Array.isArray(inputVector)) {
            throw new PipelineError(
                `NeuralHandler: input must be an array, got ${typeof inputVector}.`,
                { stepId: step.id, stepType: step.type }
            );
        }

        const inputTensor = new Tensor(new Float32Array(inputVector), [1, inputVector.length]);
        const prediction = model.network.forward(inputTensor);
        const rawOutput = Array.from(prediction.data);

        // Generic structured output parsing from step.outputs config
        if (step.outputs && typeof step.outputs === 'object') {
            const result = {};
            let offset = 0;

            for (const [key, spec] of Object.entries(step.outputs)) {
                if (Array.isArray(spec)) {
                    // Categorical: array of labels -> argmax over next spec.length values
                    const slice = rawOutput.slice(offset, offset + spec.length);
                    if (slice.length !== spec.length) {
                        throw new PipelineError(
                            `NeuralHandler: expected ${spec.length} output values for '${key}' at offset ${offset}, ` +
                            `but prediction only has ${rawOutput.length} total values.`,
                            { stepId: step.id, stepType: step.type }
                        );
                    }
                    let maxIdx = 0;
                    for (let i = 1; i < slice.length; i++) {
                        if (slice[i] > slice[maxIdx]) maxIdx = i;
                    }
                    result[key] = spec[maxIdx];
                    offset += spec.length;
                } else if (spec === 'scalar' || spec === 'float') {
                    // Single scalar value
                    result[key] = rawOutput[offset];
                    offset += 1;
                } else if (typeof spec === 'number') {
                    // Take next `spec` raw values
                    result[key] = rawOutput.slice(offset, offset + spec);
                    offset += spec;
                }
            }

            return result;
        }

        return rawOutput;
    }
}

export const neuralHandler = new NeuralHandler();
