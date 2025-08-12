/**
 * @file slmnet/ContextNavigator.js
 * @description Вспомогательная функция для безопасного извлечения данных из контекста по строковому пути.
 */

/**
 * Извлекает значение из объекта по строковому пути (например, 'a.b.c').
 * @param {object} context - Объект контекста.
 * @param {string} path - Строковый путь.
 * @returns {any} Найденное значение или undefined, если путь не существует.
 */
export function getValueFromContext(context, path) {
    // Разделяем путь на части
    const keys = path.split('.');
    let currentValue = context;

    // Последовательно "спускаемся" по объекту
    for (const key of keys) {
        if (currentValue === null || typeof currentValue !== 'object' || !key in currentValue) {
            return undefined; // Если на каком-то этапе путь прерывается, возвращаем undefined
        }
        currentValue = currentValue[key];
    }

    return currentValue;
}