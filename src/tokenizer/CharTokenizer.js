/**
 * @file src/tokenizer/CharTokenizer.js
 * @description Character-level tokenizer.
 */

class CharTokenizer {
    /**
     * @param {string[]} vocab - Sorted array of characters.
     * @param {object} [options]
     * @param {string} [options.unkToken] - Character used for unknown tokens.
     */
    constructor(vocab, { unkToken = null } = {}) {
        this.vocab = vocab;
        this.vocabSize = vocab.length;
        this.charToId = new Map();
        this.idToChar = new Map();
        vocab.forEach((ch, i) => {
            this.charToId.set(ch, i);
            this.idToChar.set(i, ch);
        });
        this.unkId = unkToken !== null ? (this.charToId.get(unkToken) ?? null) : null;
    }

    /**
     * Build tokenizer from raw text.
     * @param {string} text
     * @returns {CharTokenizer}
     */
    static fromText(text) {
        const vocab = [...new Set(text.split(''))].sort();
        return new CharTokenizer(vocab);
    }

    /**
     * Build tokenizer from serialized config.
     * @param {object} config - { vocab: string[], unkToken?: string }
     * @returns {CharTokenizer}
     */
    static fromConfig(config) {
        return new CharTokenizer(config.vocab, { unkToken: config.unkToken });
    }

    /**
     * Encode text to token IDs.
     * Unknown characters map to unkId (if set) or are skipped.
     * @param {string} text
     * @returns {number[]}
     */
    encode(text) {
        const ids = [];
        for (const ch of text) {
            const id = this.charToId.get(ch);
            if (id !== undefined) {
                ids.push(id);
            } else if (this.unkId !== null) {
                ids.push(this.unkId);
            }
        }
        return ids;
    }

    /**
     * Decode token IDs back to text.
     * @param {number[]} ids
     * @returns {string}
     */
    decode(ids) {
        let text = '';
        for (const id of ids) {
            const ch = this.idToChar.get(id);
            if (ch !== undefined) text += ch;
        }
        return text;
    }
}

export { CharTokenizer };
