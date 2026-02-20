import { describe, it, expect } from 'vitest';
import { logicGateHandler } from '../../src/pipeline/handlers/LogicGateHandler.js';

describe('LogicGateHandler', () => {
    it('should match a simple threshold rule', async () => {
        const step = {
            id: 'gate1',
            type: 'logic_gate',
            input: 'score',
            rules: { high: '> 0.8', low: 'default' },
        };
        expect(await logicGateHandler.process(step, { score: 0.9 })).toBe('high');
        expect(await logicGateHandler.process(step, { score: 0.5 })).toBe('low');
    });

    it('should support all comparison operators', async () => {
        const makeStep = (rule) => ({
            id: 'g',
            type: 'logic_gate',
            input: 'v',
            rules: { yes: rule, no: 'default' },
        });

        expect(await logicGateHandler.process(makeStep('> 5'), { v: 6 })).toBe('yes');
        expect(await logicGateHandler.process(makeStep('>= 5'), { v: 5 })).toBe('yes');
        expect(await logicGateHandler.process(makeStep('< 5'), { v: 3 })).toBe('yes');
        expect(await logicGateHandler.process(makeStep('<= 5'), { v: 5 })).toBe('yes');
        expect(await logicGateHandler.process(makeStep('== 5'), { v: 5 })).toBe('yes');
    });

    it('should fall through to default if no rule matches', async () => {
        const step = {
            id: 'g',
            type: 'logic_gate',
            input: 'v',
            rules: { high: '> 100', fallback: 'default' },
        };
        expect(await logicGateHandler.process(step, { v: 1 })).toBe('fallback');
    });

    it('should throw if no rule matches and no default', async () => {
        const step = {
            id: 'g',
            type: 'logic_gate',
            input: 'v',
            rules: { high: '> 100' },
        };
        await expect(logicGateHandler.process(step, { v: 1 })).rejects.toThrow('no rule matched');
    });

    it('should throw if input is not a number', async () => {
        const step = {
            id: 'g',
            type: 'logic_gate',
            input: 'v',
            rules: { x: '> 0' },
        };
        await expect(logicGateHandler.process(step, { v: 'hello' })).rejects.toThrow('number');
    });

    it('should throw if input or rules are missing', async () => {
        await expect(
            logicGateHandler.process({ id: 'g', type: 'logic_gate' }, {})
        ).rejects.toThrow('required');
    });

    it('should not iterate prototype properties (Object.entries)', async () => {
        const rules = Object.create({ inherited: '> 0' });
        rules.low = '< 0.5';
        rules.high = 'default';

        const step = { id: 'g', type: 'logic_gate', input: 'v', rules };
        // Value 0.1 matches 'low' (< 0.5). Should NOT match inherited '> 0'
        const result = await logicGateHandler.process(step, { v: 0.1 });
        expect(result).toBe('low');
    });
});
