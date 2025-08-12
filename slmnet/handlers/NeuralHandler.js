/**
 * @file slmnet/handlers/NeuralHandler.js
 * @description Обработчик конвейера для выполнения предсказаний на нейросети slmnet.
 */

import { Loader } from '../Loader.js';
import { Tensor } from '../Tensor.js';
import { Ops } from '../Ops.js';
import { getValueFromContext } from '../ContextNavigator.js'; // <-- ИМПОРТ

class SlmNetModel {
    constructor(modelConfig) {
        if (!modelConfig.architecture || !modelConfig.weights) {
            throw new Error("Некорректный формат модели. Отсутствуют 'architecture' или 'weights'.");
        }
        
        const arch = modelConfig.architecture;
        let weights = modelConfig.weights;

        if (!weights.ih || weights.ih.length === 0) {
            console.warn(`[SlmNetModel] Веса 'ih' не найдены. Генерация случайных весов для холодного запуска.`);
            weights.ih = Array.from({ length: arch.hidden * arch.input }, () => Math.random() * 2 - 1);
            weights.ho = Array.from({ length: arch.output * arch.hidden }, () => Math.random() * 2 - 1);
            weights.bh = Array.from({ length: arch.hidden }, () => Math.random() * 2 - 1);
            weights.bo = Array.from({ length: arch.output }, () => Math.random() * 2 - 1);
        }

        this.weights_ih = new Tensor(weights.ih, [arch.hidden, arch.input]);
        this.bias_h = new Tensor(weights.bh, [1, arch.hidden]);
        this.weights_ho = new Tensor(weights.ho, [arch.output, arch.hidden]);
        this.bias_o = new Tensor(weights.bo, [1, arch.output]);
    }

    /**
     * Выполняет предсказание (прямое распространение).
     * @param {number[]} inputArray - Входной вектор.
     * @returns {Float32Array} Выходной вектор предсказаний.
     */
    predict(inputArray) {
        let inputTensor = new Tensor(inputArray, [1, inputArray.length]);

        // **ИСПРАВЛЕНИЕ ЛОГИКИ ТРАНСПОНИРОВАНИЯ**
        // Правильный способ транспонировать матрицу [rows, cols] в [cols, rows]
        const transpose = (tensor) => {
            const [rows, cols] = tensor.shape;
            const transposedData = new Float32Array(rows * cols);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    transposedData[j * rows + i] = tensor.data[i * cols + j];
                }
            }
            return new Tensor(transposedData, [cols, rows]);
        };
        
        const weights_ih_t = transpose(this.weights_ih);
        const weights_ho_t = transpose(this.weights_ho);
        
        let hidden = Ops.matMul(inputTensor, weights_ih_t);
        hidden = Ops.add(hidden, this.bias_h);
        hidden = Ops.sigmoid(hidden);
        
        let output = Ops.matMul(hidden, weights_ho_t);
        output = Ops.add(output, this.bias_o);
        output = Ops.sigmoid(output);
        
        return output.data;
    }
}


const modelCache = new Map();

async function getModel(modelUrl) {
    if (modelCache.has(modelUrl)) {
        return modelCache.get(modelUrl);
    }
    const modelConfig = await Loader.loadJson(modelUrl);
    const model = new SlmNetModel(modelConfig);
    modelCache.set(modelUrl, model);
    return model;
}

class NeuralHandler {
    async process(step, context) {
        if (!step.model_url || !step.input) {
            throw new Error(`Для шага 'neural_model' (id: ${step.id}) должны быть указаны 'model_url' и 'input'.`);
        }
        const model = await getModel(step.model_url);
         const inputVector = getValueFromContext(context, step.input);
        if (!Array.isArray(inputVector)) {
            throw new Error(`Входные данные для шага 'neural_model' (id: ${step.id}) должны быть массивом, получено ${typeof inputVector}.`);
        }
        const prediction = model.predict(inputVector);

        if (step.outputs && Array.isArray(step.outputs.topic)) {
             const topicVector = Array.from(prediction.slice(0, step.outputs.topic.length));
             const sentimentValue = prediction[step.outputs.topic.length];
             const spamValue = prediction[step.outputs.topic.length + 1];
             let maxIndex = 0;
             topicVector.forEach((val, i) => {
                 if (val > topicVector[maxIndex]) maxIndex = i;
             });
             return {
                 topic: step.outputs.topic[maxIndex],
                 sentiment: sentimentValue,
                 spam: spamValue
             };
        }
        return Array.from(prediction);
    }
}

export const neuralHandler = new NeuralHandler();