/**
 * @file src/errors.js
 * @description Custom error hierarchy for slmnet.
 * All errors extend SlmnetError for easy catching.
 */

class SlmnetError extends Error {
    /**
     * @param {string} message
     * @param {object} [details] - Extra context for debugging.
     */
    constructor(message, details = {}) {
        super(message);
        this.name = 'SlmnetError';
        this.details = details;
    }
}

class ShapeError extends SlmnetError {
    /**
     * @param {string} message
     * @param {object} [info]
     * @param {string} [info.expected] - Expected shape description.
     * @param {string} [info.actual] - Actual shape description.
     * @param {string} [info.operation] - Name of the operation that failed.
     */
    constructor(message, { expected, actual, operation } = {}) {
        const parts = [message];
        if (expected) parts.push(`  Expected: [${expected}]`);
        if (actual) parts.push(`  Got: [${actual}]`);
        if (operation) parts.push(`  During: ${operation}`);
        super(parts.join('\n'), { expected, actual, operation });
        this.name = 'ShapeError';
    }
}

class ModelLoadError extends SlmnetError {
    /**
     * @param {string} message
     * @param {object} [info]
     * @param {string} [info.url]
     * @param {number} [info.status]
     */
    constructor(message, { url, status } = {}) {
        const parts = [message];
        if (url) parts.push(`  URL: ${url}`);
        if (status) parts.push(`  HTTP status: ${status}`);
        super(parts.join('\n'), { url, status });
        this.name = 'ModelLoadError';
    }
}

class ValidationError extends SlmnetError {
    /**
     * @param {string} message
     * @param {object} [details]
     */
    constructor(message, details = {}) {
        super(message, details);
        this.name = 'ValidationError';
    }
}

class PipelineError extends SlmnetError {
    /**
     * @param {string} message
     * @param {object} [info]
     * @param {string} [info.stepId]
     * @param {string} [info.stepType]
     * @param {Error}  [info.cause]
     */
    constructor(message, { stepId, stepType, cause } = {}) {
        const parts = [message];
        if (stepId) parts.push(`  Step: ${stepId}`);
        if (stepType) parts.push(`  Type: ${stepType}`);
        super(parts.join('\n'), { stepId, stepType, cause });
        this.name = 'PipelineError';
    }
}

export { SlmnetError, ShapeError, ModelLoadError, ValidationError, PipelineError };
