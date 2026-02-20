import { describe, it, expect, vi } from 'vitest';
import { executePipeline } from '../../src/pipeline/Executor.js';

describe('executePipeline', () => {
    it('should throw on invalid config', async () => {
        await expect(executePipeline(null, 'input')).rejects.toThrow('pipeline');
        await expect(executePipeline({}, 'input')).rejects.toThrow('pipeline');
    });

    it('should throw on unknown handler type', async () => {
        const config = {
            input_field: 'text',
            pipeline: [{ id: 'step1', type: 'nonexistent' }],
        };
        await expect(executePipeline(config, 'hello')).rejects.toThrow('No handler found');
    });

    it('should throw if step missing id or type', async () => {
        const config = {
            input_field: 'text',
            pipeline: [{ type: 'tokenizer' }], // missing id
        };
        await expect(executePipeline(config, 'hello')).rejects.toThrow("'id' and 'type'");
    });

    it('should call onProgress callbacks', async () => {
        const progressValues = [];
        const config = {
            input_field: 'text',
            pipeline: [
                { id: 'step1', type: 'logic_gate', input: 'text', rules: { yes: 'default' } },
            ],
        };
        // logic_gate operates on a value, not text. Let's set raw input to a number.
        const result = await executePipeline(
            { input_field: 'val', pipeline: [{ id: 's1', type: 'logic_gate', input: 'val', rules: { low: '< 0.5', high: 'default' } }] },
            0.3,
            { onProgress: (p) => progressValues.push(p) }
        );
        expect(progressValues).toEqual([1]);
        expect(result.s1).toBe('low');
    });
});
