import { describe, it, expect } from 'vitest';
import { decoderHandler } from '../../src/pipeline/handlers/DecoderHandler.js';
import { vectorBuilderHandler } from '../../src/pipeline/handlers/VectorBuilderHandler.js';

describe('DecoderHandler', () => {
    it('should return the category with the highest value', async () => {
        const step = { id: 'test', type: 'decoder', input: 'pred', categories: ['a', 'b', 'c'] };
        const context = { pred: [0.1, 0.9, 0.3] };
        const result = await decoderHandler.process(step, context);
        expect(result).toBe('b');
    });

    it('should throw if categories is missing', async () => {
        const step = { id: 'test', type: 'decoder', input: 'pred' };
        const context = { pred: [0.1, 0.9] };
        await expect(decoderHandler.process(step, context)).rejects.toThrow('categories');
    });

    it('should throw if input is not an array', async () => {
        const step = { id: 'test', type: 'decoder', input: 'pred', categories: ['a'] };
        const context = { pred: 'not_array' };
        await expect(decoderHandler.process(step, context)).rejects.toThrow('non-empty array');
    });

    it('should throw if argmax exceeds categories length', async () => {
        const step = { id: 'test', type: 'decoder', input: 'pred', categories: ['a'] };
        const context = { pred: [0.1, 0.9] }; // max at index 1 but only 1 category
        await expect(decoderHandler.process(step, context)).rejects.toThrow('exceeds');
    });
});

describe('VectorBuilderHandler', () => {
    it('should build one-hot encoded vector', async () => {
        const step = {
            id: 'test',
            type: 'vector_builder',
            inputs: [
                { source: 'mood', categories: ['happy', 'sad', 'neutral'] },
                { source: 'safe', categories: ['safe', 'threat'] },
            ],
        };
        const context = { mood: 'sad', safe: 'safe' };
        const result = await vectorBuilderHandler.process(step, context);
        expect(result).toEqual([0, 1, 0, 1, 0]);
    });

    it('should throw if context value is undefined', async () => {
        const step = {
            id: 'test',
            type: 'vector_builder',
            inputs: [{ source: 'missing', categories: ['a', 'b'] }],
        };
        await expect(vectorBuilderHandler.process(step, {})).rejects.toThrow('undefined');
    });

    it('should warn but not crash on unknown category value', async () => {
        const step = {
            id: 'test',
            type: 'vector_builder',
            inputs: [{ source: 'val', categories: ['a', 'b'] }],
        };
        const context = { val: 'unknown' };
        const result = await vectorBuilderHandler.process(step, context);
        expect(result).toEqual([0, 0]); // zero vector
    });
});
