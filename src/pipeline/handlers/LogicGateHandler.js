/**
 * @file src/pipeline/handlers/LogicGateHandler.js
 * @description Pipeline handler for rule-based decision logic.
 */

import { getValueFromContext } from '../ContextNavigator.js';

function evaluateRule(value, rule) {
    const parts = rule.trim().split(' ');
    if (parts.length !== 2) {
        throw new Error(`Invalid rule format: "${rule}". Expected "operator number".`);
    }
    const operator = parts[0];
    const target = parseFloat(parts[1]);
    if (isNaN(target)) {
        throw new Error(`Invalid number in rule: "${rule}".`);
    }
    switch (operator) {
        case '>':
            return value > target;
        case '>=':
            return value >= target;
        case '<':
            return value < target;
        case '<=':
            return value <= target;
        case '==':
            return value === target;
        case '===':
            return value === target;
        default:
            throw new Error(`Unknown operator: "${operator}".`);
    }
}

class LogicGateHandler {
    async process(step, context) {
        if (!step.input || !step.rules) {
            throw new Error(`Step '${step.id}': 'input' and 'rules' are required.`);
        }

        const inputValue = getValueFromContext(context, step.input);
        if (typeof inputValue !== 'number') {
            throw new Error(`Step '${step.id}': input must be a number, got ${typeof inputValue}.`);
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
                throw new Error(
                    `Step '${step.id}': no rule matched value ${inputValue} and no default set.`
                );
            }
        }

        return result;
    }
}

export const logicGateHandler = new LogicGateHandler();
