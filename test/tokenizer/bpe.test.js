import { describe, it, expect } from 'vitest';
import { BPETokenizer } from '../../src/tokenizer/BPETokenizer.js';

describe('BPETokenizer', () => {
    const vocab = { a: 0, b: 1, c: 2, ab: 3, abc: 4, ' ': 5 };
    const merges = [
        ['a', 'b'],
        ['ab', 'c'],
    ]; // a+b -> ab, ab+c -> abc

    describe('encode', () => {
        it('should apply BPE merges', () => {
            const tok = new BPETokenizer(vocab, merges);
            const ids = tok.encode('abc');
            expect(ids).toEqual([4]); // 'abc' merged fully
        });

        it('should handle multiple words', () => {
            const tok = new BPETokenizer(vocab, merges);
            const ids = tok.encode('ab c');
            // 'ab' -> merge to 'ab'(3), ' '(5), 'c'(2)
            expect(ids).toEqual([3, 5, 2]);
        });

        it('should return empty array for empty input', () => {
            const tok = new BPETokenizer(vocab, merges);
            expect(tok.encode('')).toEqual([]);
        });

        it('should skip unknown tokens by default', () => {
            const tok = new BPETokenizer(vocab, merges);
            const ids = tok.encode('x');
            expect(ids).toEqual([]); // 'x' not in vocab, skipped
        });

        it('should use unk token when configured', () => {
            const vocabWithUnk = { ...vocab, '<unk>': 6 };
            const tok = new BPETokenizer(vocabWithUnk, merges, { unkToken: '<unk>' });
            const ids = tok.encode('x');
            expect(ids).toEqual([6]);
        });
    });

    describe('decode', () => {
        it('should convert IDs back to text', () => {
            const tok = new BPETokenizer(vocab, merges);
            expect(tok.decode([3, 5, 2])).toBe('ab c');
        });
    });
});
