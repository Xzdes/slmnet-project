#!/usr/bin/env node
/**
 * @file tools/generate_sample_weights.js
 * @description Generates deterministic random weights for sample models
 *              so that demos work out of the box.
 *
 * Usage: node tools/generate_sample_weights.js
 */

import { readFileSync, writeFileSync } from 'fs';

function seededRandom(seed) {
    let s = seed;
    return () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return (s / 0x7fffffff) * 2 - 1; // range [-1, 1]
    };
}

function generateWeights(modelPath, seed) {
    const config = JSON.parse(readFileSync(modelPath, 'utf-8'));
    const { input, hidden, output } = config.architecture;
    const rand = seededRandom(seed);

    // Xavier-like initialization: scale by sqrt(2/(fan_in+fan_out))
    const scaleIH = Math.sqrt(2 / (input + hidden));
    const scaleHO = Math.sqrt(2 / (hidden + output));

    config.weights = {
        ih: Array.from({ length: hidden * input }, () => rand() * scaleIH),
        ho: Array.from({ length: output * hidden }, () => rand() * scaleHO),
        bh: Array.from({ length: hidden }, () => 0),
        bo: Array.from({ length: output }, () => 0),
    };

    // Remove comment/script fields
    delete config._comment;
    delete config._script;

    writeFileSync(modelPath, JSON.stringify(config, null, 2));
    console.log(`Generated weights for ${modelPath} (input=${input}, hidden=${hidden}, output=${output})`);
}

generateWeights('./models/experts_net.json', 42);
generateWeights('./models/director_net.json', 123);

console.log('Done.');
