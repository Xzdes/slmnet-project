/**
 * @file slmnet/slmnet.js
 * @description Главный файл библиотеки slmnet.
 * Предоставляет простой публичный API для взаимодействия с движком.
 * Это единственная точка входа для пользователя.
 */

import { executePipeline } from './Executor.js';
import { Loader } from './Loader.js';

/**
 * Основной объект slmnet, предоставляемый пользователю.
 */
const slmnet = {
    /**
     * Запускает выполнение конвейера для указанных входных данных.
     * @param {object|string} pipelineSource - Либо сам объект с конфигурацией конвейера, 
     *                                       либо URL-адрес .json файла с конфигурацией.
     * @param {any} input - Входные данные для обработки. Это может быть строка, объект и т.д.,
     *                      в зависимости от того, чего ожидает первый шаг конвейера.
     * @returns {Promise<object>} Промис, который разрешается объектом с полным контекстом выполнения,
     *                            содержащим все промежуточные и финальные результаты.
     * 
     * @example
     * // Запуск с объектом конфигурации
     * const result = await slmnet.run({ pipeline: [...] }, "Какой-то текст");
     * 
     * @example
     * // Запуск с URL-адресом конфигурации
     * const result = await slmnet.run('./pipelines/review_analysis.json', "Другой текст");
     */
    async run(pipelineSource, input) {
        try {
            let pipelineConfig;

            // 1. Определяем, передан ли нам URL или готовый объект.
            if (typeof pipelineSource === 'string') {
                // Если это строка (URL), используем наш Loader для асинхронной загрузки.
                pipelineConfig = await Loader.loadJson(pipelineSource);
            } else if (typeof pipelineSource === 'object' && pipelineSource !== null) {
                // Если это объект, используем его напрямую.
                pipelineConfig = pipelineSource;
            } else {
                // Если что-то другое, выбрасываем ошибку.
                throw new Error("Источник конвейера должен быть либо URL-строкой, либо объектом конфигурации.");
            }
            
            // 2. Вызываем ядро системы - Executor - для выполнения конвейера.
            const resultContext = await executePipeline(pipelineConfig, input);
            
            // 3. Возвращаем пользователю полный контекст с результатами.
            return resultContext;

        } catch (error) {
            console.error("[slmnet] Критическая ошибка при выполнении:", error);
            // Чтобы пользователь мог обработать ошибку в своем коде, пробрасываем ее дальше.
            throw error;
        }
    }
};

// Экспортируем наш главный объект по умолчанию.
// Это позволит пользователю писать `import slmnet from './slmnet.js'`.
export default slmnet;