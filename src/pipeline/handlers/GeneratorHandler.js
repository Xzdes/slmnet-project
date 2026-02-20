/**
 * @file src/pipeline/handlers/GeneratorHandler.js
 * @description Pipeline handler for text generation using a transformer model.
 */

import { Model } from '../../runtime/Model.js';
import { PipelineError } from '../../errors.js';
import { getValueFromContext } from '../ContextNavigator.js';

const modelCache = new Map();

async function getGeneratorModel(modelUrl) {
    if (modelCache.has(modelUrl)) return modelCache.get(modelUrl);
    const model = await Model.load(modelUrl);
    modelCache.set(modelUrl, model);
    return model;
}

class GeneratorHandler {
    async process(step, context) {
        if (!step.model_url || !step.input) {
            throw new PipelineError("GeneratorHandler requires 'model_url' and 'input'.", {
                stepId: step.id,
                stepType: step.type,
            });
        }

        const model = await getGeneratorModel(step.model_url);
        const prompt = getValueFromContext(context, step.input);

        if (typeof prompt !== 'string') {
            throw new PipelineError(
                `GeneratorHandler: input must be a string, got ${typeof prompt}.`,
                { stepId: step.id, stepType: step.type }
            );
        }

        return model.generate(prompt, {
            maxTokens: step.max_tokens || 100,
            temperature: step.temperature || 0.8,
            topK: step.top_k || 10,
            topP: step.top_p || 1.0,
        });
    }
}

export const generatorHandler = new GeneratorHandler();
