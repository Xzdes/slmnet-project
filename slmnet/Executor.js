/**
 * @file slmnet/Executor.js
 * @description Ядро системы, отвечающее за выполнение конвейера обработки.
 */

// Импортируем наши "станки" (обработчики).
import { tokenizerHandler } from './handlers/TokenizerHandler.js';
import { logicGateHandler } from './handlers/LogicGateHandler.js';
// **НОВЫЙ ИМПОРТ**
import { neuralHandler } from './handlers/NeuralHandler.js';
import { vectorBuilderHandler } from './handlers/VectorBuilderHandler.js';
import { decoderHandler } from './handlers/DecoderHandler.js';

const HandlerRegistry = new Map([
    ['tokenizer', tokenizerHandler],
    ['logic_gate', logicGateHandler],
    ['neural_model', neuralHandler],
    ['vector_builder', vectorBuilderHandler],
    ['decoder', decoderHandler]
]);


/**
 * Выполняет конвейер обработки для одного входного элемента.
 * @param {object} pipelineConfig - Полная конфигурация конвейера.
 * @param {any} rawInput - Исходные входные данные для обработки.
 * @returns {Promise<object>} Промис, который разрешается "контекстом" - объектом, 
 *                            содержащим все промежуточные и конечные результаты.
 */
export async function executePipeline(pipelineConfig, rawInput) {
    if (!pipelineConfig || !Array.isArray(pipelineConfig.pipeline)) {
        throw new Error("Некорректная конфигурация конвейера. Отсутствует поле 'pipeline'.");
    }
    
    // 1. Создаем начальный контекст выполнения.
    const context = {};
    context[pipelineConfig.input_field] = rawInput;

    // 2. Последовательно выполняем каждый шаг конвейера.
    for (const step of pipelineConfig.pipeline) {
        if (!step.id || !step.type) {
            throw new Error(`Шаг в конвейере должен иметь 'id' и 'type'. Проверьте: ${JSON.stringify(step)}`);
        }

        // 3. Находим нужный обработчик в реестре.
        const handler = HandlerRegistry.get(step.type);
        if (!handler) {
            throw new Error(`Не найден обработчик для типа "${step.type}" (id: ${step.id}).`);
        }

        // 4. Выполняем шаг.
        try {
            const stepResult = await handler.process(step, context);
            
            // 5. Сохраняем результат этого шага в контекст под его 'id'.
            context[step.id] = stepResult;
        } catch (error) {
            console.error(`Ошибка на шаге конвейера '${step.id}':`, error);
            throw error;
        }
    }
    
    // 6. Возвращаем итоговый контекст со всеми результатами.
    return context;
}