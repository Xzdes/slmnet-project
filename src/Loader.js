/**
 * @file src/Loader.js
 * @description Resource loader with caching, timeout, and TTL support.
 */

import { ModelLoadError } from './errors.js';

const DEFAULT_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
const DEFAULT_TIMEOUT = 30000; // 30 seconds

const cache = new Map();

const Loader = {
    /** Configurable defaults for caching and timeouts. */
    options: {
        cacheTTL: DEFAULT_CACHE_TTL,
        timeout: DEFAULT_TIMEOUT,
    },

    /**
     * Load and parse a JSON file. Results are cached with TTL.
     * @param {string} url
     * @param {object} [options]
     * @param {number} [options.timeout] - Override default timeout (ms).
     * @returns {Promise<object>}
     * @throws {ModelLoadError} On network or parse failure.
     */
    async loadJson(url, options = {}) {
        const cached = this._getFromCache(url);
        if (cached !== undefined) return cached;

        const response = await this._fetch(url, options);
        const data = await response.json();
        this._setCache(url, data);
        return data;
    },

    /**
     * Load a binary file as ArrayBuffer. Results are cached with TTL.
     * @param {string} url
     * @param {object} [options]
     * @param {number} [options.timeout] - Override default timeout (ms).
     * @returns {Promise<ArrayBuffer>}
     * @throws {ModelLoadError} On network failure.
     */
    async loadBinary(url, options = {}) {
        const cached = this._getFromCache(url);
        if (cached !== undefined) return cached;

        const response = await this._fetch(url, options);
        const buffer = await response.arrayBuffer();
        this._setCache(url, buffer);
        return buffer;
    },

    /**
     * Clear the entire loader cache.
     */
    clearCache() {
        cache.clear();
    },

    /** @private */
    async _fetch(url, { timeout } = {}) {
        const controller = new AbortController();
        const timeoutMs = timeout || this.options.timeout;
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const response = await fetch(url, { signal: controller.signal });
            if (!response.ok) {
                throw new ModelLoadError(
                    `Failed to load resource: ${response.statusText}`,
                    { url, status: response.status }
                );
            }
            return response;
        } catch (err) {
            if (err instanceof ModelLoadError) throw err;
            if (err.name === 'AbortError') {
                throw new ModelLoadError(
                    `Request timed out after ${timeoutMs}ms.`,
                    { url }
                );
            }
            throw new ModelLoadError(
                `Network error loading resource: ${err.message}`,
                { url }
            );
        } finally {
            clearTimeout(timeoutId);
        }
    },

    /** @private */
    _getFromCache(url) {
        const entry = cache.get(url);
        if (!entry) return undefined;
        if (Date.now() - entry.timestamp > this.options.cacheTTL) {
            cache.delete(url);
            return undefined;
        }
        return entry.data;
    },

    /** @private */
    _setCache(url, data) {
        cache.set(url, { data, timestamp: Date.now() });
    },
};

export { Loader };
