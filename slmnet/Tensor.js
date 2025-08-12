/**
 * @file slmnet/Tensor.js
 * @description Фундаментальный N-мерный контейнер данных для slmnet.
 * Это основной "атом" системы, который хранит данные и их форму (shape).
 */

class Tensor {
    /**
     * Создает новый тензор.
     * @param {Array} data - Либо плоский массив данных, либо вложенный массив.
     * @param {number[]} [shape] - Форма тензора. Если не указана, будет вычислена из вложенного массива `data`.
     */
    constructor(data, shape) {
        if (shape) {
            // Случай 1: Данные и форма предоставлены явно.
            const expectedSize = shape.reduce((acc, dim) => acc * dim, 1);
            if (data.length !== expectedSize) {
                throw new Error(`Размер данных (${data.length}) не соответствует форме [${shape}] (ожидалось ${expectedSize}).`);
            }
            this.data = new Float32Array(data);
            this.shape = shape;
        } else {
            // Случай 2: Форма не указана, вычисляем ее из вложенного массива.
            const { flatData, inferredShape } = this._inferShapeAndFlatten(data);
            this.data = new Float32Array(flatData);
            this.shape = inferredShape;
        }

        // Общее количество элементов в тензоре
        this.size = this.data.length;
    }

    /**
     * Рекурсивный хелпер для вычисления формы и "распрямления" вложенного массива.
     * @private
     * @param {Array} arr - Вложенный массив.
     * @returns {{flatData: number[], inferredShape: number[]}}
     */
    _inferShapeAndFlatten(arr) {
        const flatData = [];
        const inferredShape = [];
        
        let currentLevel = arr;
        while (Array.isArray(currentLevel)) {
            if (currentLevel.length === 0) {
                // Если массив пустой, прерываем определение формы
                break;
            }
            inferredShape.push(currentLevel.length);
            // Проверка, что все под-массивы на одном уровне имеют одинаковую длину
            if (Array.isArray(currentLevel[0])) {
                const firstSubArrayLength = currentLevel[0].length;
                for (let i = 1; i < currentLevel.length; i++) {
                    if (!Array.isArray(currentLevel[i]) || currentLevel[i].length !== firstSubArrayLength) {
                        throw new Error("Вложенные массивы имеют разную длину. Невозможно создать тензор.");
                    }
                }
            }
            currentLevel = currentLevel[0];
        }

        // Рекурсивная функция для "распрямления"
        const flatten = (subArr) => {
            for (const element of subArr) {
                if (Array.isArray(element)) {
                    flatten(element);
                } else {
                    flatData.push(element);
                }
            }
        };

        flatten(arr);
        return { flatData, inferredShape };
    }
    
    /**
     * Статический метод для создания тензора из вложенного массива.
     * Более удобный способ вызова конструктора.
     * @param {Array} arr - Вложенный массив.
     * @returns {Tensor} Новый экземпляр тензора.
     */
    static from(arr) {
        return new Tensor(arr);
    }

    /**
     * Создает тензор указанной формы, заполненный нулями.
     * @param {number[]} shape - Форма тензора.
     * @returns {Tensor} Новый тензор.
     */
    static zeros(shape) {
        const size = shape.reduce((acc, dim) => acc * dim, 1);
        const data = new Array(size).fill(0);
        return new Tensor(data, shape);
    }
    
    /**
     * Создает тензор указанной формы, заполненный случайными числами от -1 до 1.
     * @param {number[]} shape - Форма тензора.
     * @returns {Tensor} Новый тензор.
     */
    static random(shape) {
        const size = shape.reduce((acc, dim) => acc * dim, 1);
        const data = Array.from({ length: size }, () => Math.random() * 2 - 1);
        return new Tensor(data, shape);
    }
    
    /**
     * Возвращает данные тензора в виде вложенного массива.
     * Полезно для вывода и отладки.
     * @returns {Array}
     */
    toArray() {
        const buildArray = (shape, dataSlice) => {
            if (shape.length === 1) {
                return Array.from(dataSlice);
            }
            
            const newArr = [];
            const subArraySize = shape.slice(1).reduce((acc, dim) => acc * dim, 1);
            
            for (let i = 0; i < shape[0]; i++) {
                const start = i * subArraySize;
                const end = start + subArraySize;
                newArr.push(buildArray(shape.slice(1), dataSlice.slice(start, end)));
            }
            return newArr;
        };
        
        return buildArray(this.shape, this.data);
    }

    /**
     * Выводит тензор в консоль в удобном для чтения виде.
     */
    print() {
        console.log('Tensor {');
        console.log('  shape:', this.shape);
        console.log('  data:', this.toArray());
        console.log('}');
    }
}

// Экспортируем класс для использования в других модулях
export { Tensor };