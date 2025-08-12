/**
 * @file slmnet/handlers/TokenizerHandler.js
 * @description Обработчик конвейера для преобразования текста в вектор "мешка слов".
 */

import { Loader } from '../Loader.js';
import { getValueFromContext } from '../ContextNavigator.js'; // <-- ИМПОРТ

const vocabCache = new Map();

async function getVocabulary(vocabUrl) {
    if (vocabCache.has(vocabUrl)) {
        return vocabCache.get(vocabUrl);
    }
    const vocabData = await Loader.loadJson(vocabUrl);
    if (!Array.isArray(vocabData)) {
        throw new Error(`Словарь по адресу ${vocabUrl} должен быть массивом, получен ${typeof vocabData}.`);
    }
    vocabCache.set(vocabUrl, vocabData);
    return vocabData;
}

/**
 * Преобразует текст в вектор "мешка слов" (Bag of Words).
 * @private
 * @param {string} text - Входной текст.
 * @param {string[]} vocabulary - Словарь, на основе которого создается вектор.
 * @param {number} vectorSize - **Обязательный** размер выходного вектора.
 * @returns {number[]} - Вектор из 0 и 1.
 */
function textToBoW(text, vocabulary, vectorSize) {
    // **КЛЮЧЕВОЕ ИЗМЕНЕНИЕ:** Вектор создается с размером, который требует модель, а не по длине словаря.
    const vector = new Array(vectorSize).fill(0);
    
    const lowerCaseText = text.toLowerCase().replace(/[.,!?"'()`]/g, '');
    const wordsInText = new Set(lowerCaseText.split(/\s+/g));
    
    for (const word of wordsInText) {
        const index = vocabulary.indexOf(word);
        // Мы добавляем 1 только если слово найдено И его индекс не выходит за пределы требуемого размера вектора.
        if (index !== -1 && index < vectorSize) {
            vector[index] = 1;
        }
    }
    return vector;
}

class TokenizerHandler {
    async process(step, context) {
        // Проверяем наличие обязательных параметров. Теперь 'output_size' - обязательный.
        if (!step.vocab_url || !step.output_size) {
            throw new Error(`Для шага токенизации (id: ${step.id}) должны быть указаны 'vocab_url' и 'output_size'.`);
        }
        if (typeof step.output_size !== 'number') {
             throw new Error(`Параметр 'output_size' для шага (id: ${step.id}) должен быть числом.`);
        }
        const inputText = getValueFromContext(context, step.input);
        if (typeof inputText !== 'string') {
            throw new Error(`Входные данные для шага токенизации (id: ${step.id}) должны быть строкой.`);
        }

        const vocabulary = await getVocabulary(step.vocab_url);
        
        // Передаем требуемый размер в функцию-векторизатор.
        const vector = textToBoW(inputText, vocabulary, step.output_size);
        
        return vector;
    }
}

export const tokenizerHandler = new TokenizerHandler();