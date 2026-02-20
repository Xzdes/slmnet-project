import { describe, it, expect } from 'vitest';
import { SlmnetError, ShapeError, ModelLoadError, ValidationError, PipelineError } from '../src/errors.js';

describe('Error classes', () => {
    it('SlmnetError should be an instance of Error', () => {
        const err = new SlmnetError('test');
        expect(err).toBeInstanceOf(Error);
        expect(err).toBeInstanceOf(SlmnetError);
        expect(err.name).toBe('SlmnetError');
        expect(err.message).toBe('test');
    });

    it('ShapeError should extend SlmnetError and include details', () => {
        const err = new ShapeError('bad shape', {
            expected: '2, 3',
            actual: '3, 2',
            operation: 'matMul',
        });
        expect(err).toBeInstanceOf(SlmnetError);
        expect(err.name).toBe('ShapeError');
        expect(err.details.operation).toBe('matMul');
        expect(err.message).toContain('bad shape');
        expect(err.message).toContain('2, 3');
    });

    it('ModelLoadError should include url and status', () => {
        const err = new ModelLoadError('not found', { url: '/model.slmnet', status: 404 });
        expect(err).toBeInstanceOf(SlmnetError);
        expect(err.details.url).toBe('/model.slmnet');
        expect(err.details.status).toBe(404);
    });

    it('PipelineError should include step context', () => {
        const err = new PipelineError('step failed', { stepId: 'l1_report', stepType: 'neural_model' });
        expect(err).toBeInstanceOf(SlmnetError);
        expect(err.details.stepId).toBe('l1_report');
        expect(err.message).toContain('l1_report');
    });

    it('all errors should be catchable as SlmnetError', () => {
        const errors = [
            new ShapeError('a'),
            new ModelLoadError('b'),
            new ValidationError('c'),
            new PipelineError('d'),
        ];
        for (const err of errors) {
            expect(err).toBeInstanceOf(SlmnetError);
        }
    });
});
