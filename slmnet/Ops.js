/**
 * @file slmnet/Ops.js
 * @description Набор чистых математических операций для работы с тензорами.
 * Каждая функция принимает на вход один или несколько тензоров и возвращает новый тензор.
 */

import { Tensor } from './Tensor.js';

/**
 * Выполняет матричное умножение двух 2D-тензоров (матриц).
 * @param {Tensor} a - Левый тензор (матрица) с формой [rowsA, colsA].
 * @param {Tensor} b - Правый тензор (матрица) с формой [colsA, colsB].
 * @returns {Tensor} Новый тензор с формой [rowsA, colsB].
 */
function matMul(a, b) {
    // Проверка на корректность размерностей
    if (a.shape.length !== 2 || b.shape.length !== 2) {
        throw new Error('Матричное умножение поддерживается только для 2D-тензоров.');
    }
    if (a.shape[1] !== b.shape[0]) {
        throw new Error(`Несовместимые формы для матричного умножения: [${a.shape}] и [${b.shape}].`);
    }

    const [rowsA, colsA] = a.shape;
    const [rowsB, colsB] = b.shape; // rowsB === colsA

    const resultShape = [rowsA, colsB];
    const resultData = new Float32Array(rowsA * colsB);

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
                // a.data[i * colsA + k] - доступ к элементу a[i, k]
                // b.data[k * colsB + j] - доступ к элементу b[k, j]
                sum += a.data[i * colsA + k] * b.data[k * colsB + j];
            }
            // resultData[i * colsB + j] - доступ к элементу result[i, j]
            resultData[i * colsB + j] = sum;
        }
    }

    return new Tensor(resultData, resultShape);
}

/**
 * Выполняет поэлементное сложение двух тензоров.
 * Поддерживает "вещание" (broadcasting) для векторов.
 * @param {Tensor} a - Первый тензор.
 * @param {Tensor} b - Второй тензор. Может быть вектором для вещания.
 * @returns {Tensor} Новый тензор с той же формой, что и у 'a'.
 */
function add(a, b) {
    // Простой случай: формы полностью совпадают
    if (a.size === b.size && a.shape.every((dim, i) => dim === b.shape[i])) {
        const resultData = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++) {
            resultData[i] = a.data[i] + b.data[i];
        }
        return new Tensor(resultData, a.shape);
    }

    // Случай "вещания" (broadcasting): сложение матрицы [R, C] и вектора [1, C] или [C]
    if (a.shape.length === 2 && (b.shape.length === 1 || b.shape[0] === 1) && a.shape[1] === b.shape[b.shape.length - 1]) {
        const [rows, cols] = a.shape;
        const resultData = new Float32Array(a.size);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                resultData[i * cols + j] = a.data[i * cols + j] + b.data[j];
            }
        }
        return new Tensor(resultData, a.shape);
    }
    
    throw new Error(`Сложение для форм [${a.shape}] и [${b.shape}] не поддерживается.`);
}

/**
 * Применяет функцию активации ReLU к каждому элементу тензора.
 * @param {Tensor} a - Входной тензор.
 * @returns {Tensor} Новый тензор с той же формой.
 */
function relu(a) {
    const resultData = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
        resultData[i] = Math.max(0, a.data[i]);
    }
    return new Tensor(resultData, a.shape);
}

/**
 * Применяет функцию активации Sigmoid к каждому элементу тензора.
 * @param {Tensor} a - Входной тензор.
 * @returns {Tensor} Новый тензор с той же формой.
 */
function sigmoid(a) {
    const resultData = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
        resultData[i] = 1 / (1 + Math.exp(-a.data[i]));
    }
    return new Tensor(resultData, a.shape);
}

/**
 * Применяет функцию активации Softmax к 1D или 2D тензору (по строкам).
 * @param {Tensor} a - Входной тензор.
 * @returns {Tensor} Новый тензор с той же формой.
 */
function softmax(a) {
    if (a.shape.length > 2) {
        throw new Error("Softmax поддерживается только для 1D и 2D тензоров.");
    }
    
    const resultData = new Float32Array(a.size);
    const resultShape = a.shape;

    // Если это 2D тензор (батч векторов)
    if (a.shape.length === 2) {
        const [rows, cols] = a.shape;
        for (let i = 0; i < rows; i++) {
            const start = i * cols;
            const end = start + cols;
            const rowSlice = a.data.slice(start, end);

            // Для численной стабильности вычитаем максимум
            const maxVal = Math.max(...rowSlice);
            const exps = rowSlice.map(x => Math.exp(x - maxVal));
            const sumExps = exps.reduce((acc, val) => acc + val, 0);
            const softmaxRow = exps.map(e => e / sumExps);
            
            resultData.set(softmaxRow, start);
        }
    } else { // Если это 1D тензор (один вектор)
        const maxVal = Math.max(...a.data);
        const exps = Array.from(a.data).map(x => Math.exp(x - maxVal));
        const sumExps = exps.reduce((acc, val) => acc + val, 0);
        const softmaxData = exps.map(e => e / sumExps);
        resultData.set(softmaxData);
    }

    return new Tensor(resultData, resultShape);
}


// Экспортируем все операции как единый объект
const Ops = {
    matMul,
    add,
    relu,
    sigmoid,
    softmax
};

export { Ops };