/**
 * @file src/tokenizer/BPETokenizer.js
 * @description Byte-Pair Encoding tokenizer for subword tokenization.
 */

class BPETokenizer {
    /**
     * @param {Map<string, number>|object} vocab - Token string -> token ID.
     * @param {Array<[string, string]>} merges - BPE merge rules in order.
     * @param {object} [options]
     * @param {string} [options.unkToken] - Token string used for unknowns.
     */
    constructor(vocab, merges, { unkToken = null } = {}) {
        this.tokenToId = vocab instanceof Map ? vocab : new Map(Object.entries(vocab));
        this.idToToken = new Map();
        this.tokenToId.forEach((id, token) => this.idToToken.set(id, token));
        this.vocabSize = this.tokenToId.size;
        this.merges = merges;
        this.unkId = unkToken !== null ? this.tokenToId.get(unkToken) ?? null : null;

        // Build merge lookup for fast pair matching
        this.mergeRanks = new Map();
        merges.forEach(([a, b], i) => {
            this.mergeRanks.set(a + ' ' + b, i);
        });
    }

    /**
     * Build tokenizer from serialized config.
     * @param {object} config - { vocab: {token: id, ...}, merges: [[a, b], ...], unkToken?: string }
     * @returns {BPETokenizer}
     */
    static fromConfig(config) {
        return new BPETokenizer(config.vocab, config.merges, { unkToken: config.unkToken });
    }

    /**
     * Encode text to token IDs.
     * Unknown tokens map to unkId (if set) or are skipped.
     * @param {string} text
     * @returns {number[]}
     */
    encode(text) {
        if (!text) return [];

        // Pre-tokenize: split into words (simple whitespace-based)
        const words = text.match(/\S+|\s+/g) || [];
        const allIds = [];

        for (const word of words) {
            // Split word into individual characters
            let symbols = word.split('');

            // Apply BPE merges iteratively
            while (symbols.length > 1) {
                let bestPair = null;
                let bestRank = Infinity;

                for (let i = 0; i < symbols.length - 1; i++) {
                    const key = symbols[i] + ' ' + symbols[i + 1];
                    const rank = this.mergeRanks.get(key);
                    if (rank !== undefined && rank < bestRank) {
                        bestRank = rank;
                        bestPair = i;
                    }
                }

                if (bestPair === null) break;

                const merged = symbols[bestPair] + symbols[bestPair + 1];
                symbols = [
                    ...symbols.slice(0, bestPair),
                    merged,
                    ...symbols.slice(bestPair + 2),
                ];
            }

            // Convert symbols to IDs
            for (const sym of symbols) {
                const id = this.tokenToId.get(sym);
                if (id !== undefined) {
                    allIds.push(id);
                } else if (this.unkId !== null) {
                    allIds.push(this.unkId);
                }
            }
        }

        return allIds;
    }

    /**
     * Decode token IDs back to text.
     * @param {number[]} ids
     * @returns {string}
     */
    decode(ids) {
        let text = '';
        for (const id of ids) {
            const token = this.idToToken.get(id);
            if (token !== undefined) text += token;
        }
        return text;
    }
}

export { BPETokenizer };
