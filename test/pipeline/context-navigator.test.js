import { describe, it, expect } from 'vitest';
import { getValueFromContext } from '../../src/pipeline/ContextNavigator.js';

describe('ContextNavigator', () => {
    it('should get top-level value', () => {
        expect(getValueFromContext({ foo: 42 }, 'foo')).toBe(42);
    });

    it('should get nested value via dot notation', () => {
        const ctx = { a: { b: { c: 'deep' } } };
        expect(getValueFromContext(ctx, 'a.b.c')).toBe('deep');
    });

    it('should return undefined for missing path', () => {
        expect(getValueFromContext({ a: 1 }, 'b')).toBeUndefined();
        expect(getValueFromContext({ a: 1 }, 'a.b.c')).toBeUndefined();
    });

    it('should handle null/undefined in path', () => {
        expect(getValueFromContext({ a: null }, 'a.b')).toBeUndefined();
        expect(getValueFromContext({ a: undefined }, 'a.b')).toBeUndefined();
    });
});
