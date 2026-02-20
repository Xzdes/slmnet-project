/**
 * @file src/pipeline/handlers/DecoderHandler.js
 * @description Pipeline handler for decoding prediction vectors to labels (argmax).
 */

import { PipelineError } from '../../errors.js';
import { getValueFromContext } from '../ContextNavigator.js';

class DecoderHandler {
    async process(step, context) {
        if (!step.input) {
            throw new PipelineError(
                "DecoderHandler requires an 'input' field.",
                { stepId: step.id, stepType: step.type }
            );
        }
        if (!Array.isArray(step.categories) || step.categories.length === 0) {
            throw new PipelineError(
                "DecoderHandler requires a non-empty 'categories' array.",
                { stepId: step.id, stepType: step.type }
            );
        }

        const predictionVector = getValueFromContext(context, step.input);

        if (!Array.isArray(predictionVector) || predictionVector.length === 0) {
            throw new PipelineError(
                `DecoderHandler: input '${step.input}' must be a non-empty array, got ${typeof predictionVector}.`,
                { stepId: step.id, stepType: step.type }
            );
        }

        let maxIndex = 0;
        for (let i = 1; i < predictionVector.length; i++) {
            if (predictionVector[i] > predictionVector[maxIndex]) {
                maxIndex = i;
            }
        }

        if (maxIndex >= step.categories.length) {
            throw new PipelineError(
                `DecoderHandler: argmax index ${maxIndex} exceeds categories length ${step.categories.length}. ` +
                `Prediction vector has ${predictionVector.length} elements but only ${step.categories.length} categories are defined.`,
                { stepId: step.id, stepType: step.type }
            );
        }

        return step.categories[maxIndex];
    }
}

export const decoderHandler = new DecoderHandler();
