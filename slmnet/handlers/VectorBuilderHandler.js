/**
 * @file slmnet/handlers/VectorBuilderHandler.js
 * @description Собирает итоговый вектор из разных частей контекста для подачи в DirectorNet.
 */

import { getValueFromContext } from '../ContextNavigator.js'; // <-- ИМПОРТ
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index > -1) encoding[index] = 1;
    return encoding;
}

class VectorBuilderHandler {
    async process(step, context) {
        let finalVector = [];
        for (const part of step.inputs) {
            // ИЗМЕНЕНИЕ
            const value = getValueFromContext(context, part.source);
            const encodedPart = oneHotEncode(value, part.categories);
            finalVector = finalVector.concat(encodedPart);
        }
        return finalVector;
    }
}
export const vectorBuilderHandler = new VectorBuilderHandler();