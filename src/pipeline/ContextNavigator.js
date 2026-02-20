/**
 * @file src/pipeline/ContextNavigator.js
 * @description Safely extract values from context by dot-notation path.
 */

/**
 * @param {object} context
 * @param {string} path - e.g. 'l1_report.sentiment'
 * @returns {any}
 */
export function getValueFromContext(context, path) {
    const keys = path.split('.');
    let current = context;
    for (const key of keys) {
        if (current === null || current === undefined || typeof current !== 'object') {
            return undefined;
        }
        current = current[key];
    }
    return current;
}
