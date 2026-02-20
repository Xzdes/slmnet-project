import { describe, it, expect } from 'vitest';
import {
    ModelFormat,
    ARCH_TYPE,
    QUANT_TYPE,
    TOKENIZER_TYPE,
} from '../../src/runtime/ModelFormat.js';

describe('ModelFormat', () => {
    function makeTestModel(quantization = QUANT_TYPE.FLOAT32) {
        return {
            header: {
                version: 1,
                archType: ARCH_TYPE.MLP,
                quantization,
                vocabSize: 10,
                embedDim: 4,
                numHeads: 0,
                numLayers: 0,
                blockSize: 0,
                hiddenDim: 8,
                tokenizerType: TOKENIZER_TYPE.CHAR,
                numLabels: 2,
            },
            tokenizerConfig: { vocab: ['a', 'b', 'c'] },
            labels: ['positive', 'negative'],
            weights: new Map([
                ['layer_0.weights', { shape: [10, 8], data: new Float32Array(80).fill(0.1) }],
                ['layer_0.bias', { shape: [8], data: new Float32Array(8).fill(0.01) }],
            ]),
        };
    }

    describe('float32 roundtrip', () => {
        it('should parse/build roundtrip correctly', () => {
            const original = makeTestModel();
            const buffer = ModelFormat.build(original);
            const parsed = ModelFormat.parse(buffer);

            expect(parsed.header.vocabSize).toBe(10);
            expect(parsed.header.archType).toBe(ARCH_TYPE.MLP);
            expect(parsed.labels).toEqual(['positive', 'negative']);
            expect(parsed.tokenizerConfig.vocab).toEqual(['a', 'b', 'c']);

            const w = parsed.weights.get('layer_0.weights');
            expect(w.shape).toEqual([10, 8]);
            expect(Math.abs(w.data[0] - 0.1)).toBeLessThan(1e-6);

            const b = parsed.weights.get('layer_0.bias');
            expect(b.shape).toEqual([8]);
        });
    });

    describe('int8 roundtrip', () => {
        it('should quantize/dequantize with reasonable precision', () => {
            const original = makeTestModel(QUANT_TYPE.INT8);
            // Use a wider range of values
            const wData = new Float32Array(80);
            for (let i = 0; i < 80; i++) wData[i] = (i - 40) / 40; // range [-1, 1]
            original.weights.get('layer_0.weights').data = wData;

            const buffer = ModelFormat.build(original);
            const parsed = ModelFormat.parse(buffer);

            const w = parsed.weights.get('layer_0.weights');
            expect(w.shape).toEqual([10, 8]);

            // Int8 quantization loses some precision, but should be within ~1%
            for (let i = 0; i < 80; i++) {
                expect(Math.abs(w.data[i] - wData[i])).toBeLessThan(0.02);
            }
        });
    });

    describe('error handling', () => {
        it('should throw on invalid magic number', () => {
            const buffer = new ArrayBuffer(64);
            expect(() => ModelFormat.parse(buffer)).toThrow('bad magic');
        });
    });
});
