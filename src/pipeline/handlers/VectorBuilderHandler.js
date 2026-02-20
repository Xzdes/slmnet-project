/**
 * @file src/pipeline/handlers/VectorBuilderHandler.js
 * @description Pipeline handler for assembling feature vectors from context parts.
 */

import { PipelineError } from '../../errors.js';
import { getValueFromContext } from '../ContextNavigator.js';

function oneHotEncode(value, categories, stepId) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index === -1) {
        // eslint-disable-next-line no-console
        console.warn(
            `[slmnet] VectorBuilder step '${stepId}': value "${value}" not found in categories [${categories.join(', ')}]. Using zero vector.`
        );
    } else {
        encoding[index] = 1;
    }
    return encoding;
}

class VectorBuilderHandler {
    async process(step, context) {
        if (!Array.isArray(step.inputs) || step.inputs.length === 0) {
            throw new PipelineError(
                "VectorBuilderHandler requires a non-empty 'inputs' array.",
                { stepId: step.id, stepType: step.type }
            );
        }

        let finalVector = [];
        for (const part of step.inputs) {
            if (!part.source) {
                throw new PipelineError(
                    "VectorBuilder input part requires a 'source' field.",
                    { stepId: step.id, stepType: step.type }
                );
            }
            if (!Array.isArray(part.categories) || part.categories.length === 0) {
                throw new PipelineError(
                    "VectorBuilder input part requires a non-empty 'categories' array.",
                    { stepId: step.id, stepType: step.type }
                );
            }

            const value = getValueFromContext(context, part.source);
            if (value === undefined) {
                throw new PipelineError(
                    `VectorBuilder: context path '${part.source}' resolved to undefined. ` +
                    `Check that a previous pipeline step produces this value.`,
                    { stepId: step.id, stepType: step.type }
                );
            }

            const encoded = oneHotEncode(value, part.categories, step.id);
            finalVector = finalVector.concat(encoded);
        }
        return finalVector;
    }
}

export const vectorBuilderHandler = new VectorBuilderHandler();
