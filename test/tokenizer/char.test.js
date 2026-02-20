import { describe, it, expect } from 'vitest';
import { CharTokenizer } from '../../src/tokenizer/CharTokenizer.js';

describe('CharTokenizer', () => {
    describe('fromText', () => {
        it('should build vocab from text', () => {
            const tok = CharTokenizer.fromText('abcba');
            expect(tok.vocabSize).toBe(3); // a, b, c
        });
    });

    describe('encode/decode', () => {
        it('should roundtrip encode and decode', () => {
            const tok = new CharTokenizer(['a', 'b', 'c']);
            const ids = tok.encode('abc');
            expect(ids).toEqual([0, 1, 2]);
            const text = tok.decode(ids);
            expect(text).toBe('abc');
        });

        it('should skip unknown chars by default', () => {
            const tok = new CharTokenizer(['a', 'b']);
            const ids = tok.encode('axb');
            expect(ids).toEqual([0, 1]); // 'x' skipped
        });

        it('should use unk token when configured', () => {
            const tok = new CharTokenizer(['a', 'b', '?'], { unkToken: '?' });
            const ids = tok.encode('axb');
            expect(ids).toEqual([0, 2, 1]); // 'x' -> unk(?) -> 2
        });
    });

    describe('fromConfig', () => {
        it('should restore from config with unkToken', () => {
            const tok = CharTokenizer.fromConfig({ vocab: ['a', 'b', '<unk>'], unkToken: '<unk>' });
            expect(tok.unkId).toBe(2);
        });
    });
});
