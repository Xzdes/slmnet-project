/**
 * @file slmnet/handlers/DecoderHandler.js
 * @description Декодирует финальный вектор предсказаний в читаемый вердикт.
 */

import { getValueFromContext } from '../ContextNavigator.js'; // <-- ИМПОРТ
class DecoderHandler {
    async process(step, context) {
        const predictionVector = getValueFromContext(context, step.input);
        let maxIndex = 0;
        predictionVector.forEach((val, i) => {
            if (val > predictionVector[maxIndex]) {
                maxIndex = i;
            }
        });
        return step.categories[maxIndex];
    }
}
export const decoderHandler = new DecoderHandler();