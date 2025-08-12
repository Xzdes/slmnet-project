/**
 * @file slmnet/handlers/LogicGateHandler.js
 * @description Обработчик конвейера для принятия решений на основе простых правил.
 */
import { getValueFromContext } from '../ContextNavigator.js'; // <-- НОВЫЙ ИМПОРТ
/**
 * Проверяет, удовлетворяет ли значение заданному правилу.
 * @private
 * @param {number} value - Входное числовое значение.
 * @param {string} rule - Правило в формате "оператор число" (например, "> 0.7" или "<= 0.3").
 * @returns {boolean}
 */
function evaluateRule(value, rule) {
    const parts = rule.trim().split(' ');
    if (parts.length !== 2) {
        throw new Error(`Некорректный формат правила: "${rule}". Ожидался "оператор число".`);
    }
    
    const operator = parts[0];
    const targetValue = parseFloat(parts[1]);

    if (isNaN(targetValue)) {
        throw new Error(`Некорректное число в правиле: "${rule}".`);
    }

    switch (operator) {
        case '>': return value > targetValue;
        case '>=': return value >= targetValue;
        case '<': return value < targetValue;
        case '<=': return value <= targetValue;
        case '==': return value == targetValue;
        case '===': return value === targetValue;
        default:
            throw new Error(`Неизвестный оператор в правиле: "${operator}".`);
    }
}

/**
 * Класс-обработчик для этапа логических правил.
 */
class LogicGateHandler {
    async process(step, context) {
        if (!step.input || !step.rules) {
            throw new Error(`Для шага 'logic_gate' (id: ${step.id}) должны быть указаны 'input' и 'rules'.`);
        }

        // **КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Используем навигатор для получения данных**
        const inputValue = getValueFromContext(context, step.input);

        if (typeof inputValue !== 'number') {
            throw new Error(`Входные данные для шага 'logic_gate' (id: ${step.id}) должны быть числом, получено ${typeof inputValue} (путь: ${step.input}).`);
        }

        let result = null;
        let defaultResult = null;

        for (const key in step.rules) {
            const rule = step.rules[key];
            if (rule === 'default') {
                defaultResult = key;
                continue;
            }
            if (evaluateRule(inputValue, rule)) {
                result = key;
                break;
            }
        }
        
        if (result === null) {
            if (defaultResult !== null) {
                result = defaultResult;
            } else {
                throw new Error(`Ни одно из правил не подошло для значения ${inputValue} и не указано правило 'default' (id: ${step.id}).`);
            }
        }
        
        return result;
    }
}

export const logicGateHandler = new LogicGateHandler();