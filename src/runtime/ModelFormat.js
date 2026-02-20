/**
 * @file src/runtime/ModelFormat.js
 * @description Parser for the .slmnet binary model format.
 *
 * Format specification:
 *
 * [HEADER - 64 bytes]
 *   magic:          4 bytes   "SLMN"
 *   version:        4 bytes   uint32
 *   archType:       4 bytes   uint32 (0=mlp, 1=transformer)
 *   quantization:   4 bytes   uint32 (0=float32, 1=int8)
 *   vocabSize:      4 bytes   uint32
 *   embedDim:       4 bytes   uint32
 *   numHeads:       4 bytes   uint32
 *   numLayers:      4 bytes   uint32
 *   blockSize:      4 bytes   uint32
 *   hiddenDim:      4 bytes   uint32
 *   tokenizerType:  4 bytes   uint32 (0=char, 1=bpe, 2=bow)
 *   numLabels:      4 bytes   uint32
 *   reserved:       16 bytes
 *
 * [TOKENIZER SECTION]
 *   tokenizerSize:  4 bytes   uint32
 *   tokenizerData:  variable  JSON-encoded
 *
 * [LABEL SECTION] (if numLabels > 0)
 *   labelsSize:     4 bytes   uint32
 *   labelsData:     variable  JSON-encoded string[]
 *
 * [WEIGHTS SECTION]
 *   numTensors:     4 bytes   uint32
 *   For each tensor:
 *     nameLen:      4 bytes   uint32
 *     name:         variable  UTF-8
 *     ndims:        4 bytes   uint32
 *     shape:        ndims*4   uint32[]
 *     data:         product(shape)*4  float32[] (or *1 for int8)
 */

const MAGIC = 0x4E4D4C53; // "SLMN" in little-endian
const HEADER_SIZE = 64;

const ARCH_TYPE = { MLP: 0, TRANSFORMER: 1 };
const QUANT_TYPE = { FLOAT32: 0, INT8: 1 };
const TOKENIZER_TYPE = { CHAR: 0, BPE: 1, BOW: 2 };

class ModelFormat {
    /**
     * Parse a .slmnet binary buffer.
     * @param {ArrayBuffer} buffer
     * @returns {object} Parsed model data.
     */
    static parse(buffer) {
        const view = new DataView(buffer);
        let offset = 0;

        // --- HEADER ---
        const magic = view.getUint32(0, true);
        if (magic !== MAGIC) {
            throw new Error(`Invalid .slmnet file: bad magic (got 0x${magic.toString(16)}).`);
        }

        const header = {
            version:       view.getUint32(4, true),
            archType:      view.getUint32(8, true),
            quantization:  view.getUint32(12, true),
            vocabSize:     view.getUint32(16, true),
            embedDim:      view.getUint32(20, true),
            numHeads:      view.getUint32(24, true),
            numLayers:     view.getUint32(28, true),
            blockSize:     view.getUint32(32, true),
            hiddenDim:     view.getUint32(36, true),
            tokenizerType: view.getUint32(40, true),
            numLabels:     view.getUint32(44, true),
        };
        offset = HEADER_SIZE;

        // --- TOKENIZER SECTION ---
        const tokenizerSize = view.getUint32(offset, true);
        offset += 4;
        const tokenizerBytes = new Uint8Array(buffer, offset, tokenizerSize);
        const tokenizerConfig = JSON.parse(new TextDecoder().decode(tokenizerBytes));
        offset += tokenizerSize;

        // --- LABEL SECTION ---
        let labels = null;
        if (header.numLabels > 0) {
            const labelsSize = view.getUint32(offset, true);
            offset += 4;
            const labelsBytes = new Uint8Array(buffer, offset, labelsSize);
            labels = JSON.parse(new TextDecoder().decode(labelsBytes));
            offset += labelsSize;
        }

        // --- WEIGHTS SECTION ---
        const numTensors = view.getUint32(offset, true);
        offset += 4;

        const weights = new Map();
        const isInt8 = header.quantization === QUANT_TYPE.INT8;

        for (let t = 0; t < numTensors; t++) {
            // Name
            const nameLen = view.getUint32(offset, true);
            offset += 4;
            const nameBytes = new Uint8Array(buffer, offset, nameLen);
            const name = new TextDecoder().decode(nameBytes);
            offset += nameLen;

            // Shape
            const ndims = view.getUint32(offset, true);
            offset += 4;
            const shape = [];
            for (let d = 0; d < ndims; d++) {
                shape.push(view.getUint32(offset, true));
                offset += 4;
            }

            // Data
            const totalElements = shape.reduce((a, b) => a * b, 1);
            let data;

            if (isInt8) {
                // Quantized: read int8 + scale factor
                const scale = view.getFloat32(offset, true);
                offset += 4;
                const int8Data = new Int8Array(buffer, offset, totalElements);
                offset += totalElements;
                // Align to 4 bytes
                offset = (offset + 3) & ~3;

                // Dequantize to float32
                data = new Float32Array(totalElements);
                for (let i = 0; i < totalElements; i++) {
                    data[i] = int8Data[i] * scale;
                }
            } else {
                // Float32
                data = new Float32Array(buffer.slice(offset, offset + totalElements * 4));
                offset += totalElements * 4;
            }

            weights.set(name, { shape, data });
        }

        return { header, tokenizerConfig, labels, weights };
    }

    /**
     * Build a .slmnet binary buffer from model data.
     * @param {object} params
     * @returns {ArrayBuffer}
     */
    static build({ header, tokenizerConfig, labels, weights }) {
        const encoder = new TextEncoder();

        // Encode tokenizer and labels as JSON
        const tokenizerBytes = encoder.encode(JSON.stringify(tokenizerConfig));
        const labelsBytes = labels ? encoder.encode(JSON.stringify(labels)) : null;

        const isInt8 = (header.quantization || 0) === QUANT_TYPE.INT8;

        // Calculate total size
        let totalSize = HEADER_SIZE;
        totalSize += 4 + tokenizerBytes.length; // tokenizer section
        if (labelsBytes) totalSize += 4 + labelsBytes.length; // labels section
        totalSize += 4; // numTensors

        const weightEntries = [...weights.entries()];
        for (const [name, { shape, data }] of weightEntries) {
            const nameBytes = encoder.encode(name);
            totalSize += 4 + nameBytes.length;       // name
            totalSize += 4 + shape.length * 4;       // ndims + shape
            if (isInt8) {
                totalSize += 4;                      // scale factor
                const numElements = data.length || shape.reduce((a, b) => a * b, 1);
                totalSize += numElements;            // int8 data
                // Align absolute offset to 4 bytes
                totalSize = (totalSize + 3) & ~3;
            } else {
                totalSize += data.length * 4;        // float32 data
            }
        }

        // Allocate buffer
        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);
        let offset = 0;

        // Write header
        view.setUint32(0, MAGIC, true);
        view.setUint32(4, header.version || 1, true);
        view.setUint32(8, header.archType, true);
        view.setUint32(12, header.quantization || 0, true);
        view.setUint32(16, header.vocabSize, true);
        view.setUint32(20, header.embedDim || 0, true);
        view.setUint32(24, header.numHeads || 0, true);
        view.setUint32(28, header.numLayers || 0, true);
        view.setUint32(32, header.blockSize || 0, true);
        view.setUint32(36, header.hiddenDim || 0, true);
        view.setUint32(40, header.tokenizerType || 0, true);
        view.setUint32(44, header.numLabels || 0, true);
        offset = HEADER_SIZE;

        // Write tokenizer section
        view.setUint32(offset, tokenizerBytes.length, true);
        offset += 4;
        new Uint8Array(buffer, offset, tokenizerBytes.length).set(tokenizerBytes);
        offset += tokenizerBytes.length;

        // Write labels section
        if (labelsBytes) {
            view.setUint32(offset, labelsBytes.length, true);
            offset += 4;
            new Uint8Array(buffer, offset, labelsBytes.length).set(labelsBytes);
            offset += labelsBytes.length;
        }

        // Write weights section
        view.setUint32(offset, weightEntries.length, true);
        offset += 4;

        for (const [name, { shape, data }] of weightEntries) {
            const nameBytes = encoder.encode(name);

            view.setUint32(offset, nameBytes.length, true);
            offset += 4;
            new Uint8Array(buffer, offset, nameBytes.length).set(nameBytes);
            offset += nameBytes.length;

            view.setUint32(offset, shape.length, true);
            offset += 4;
            for (const dim of shape) {
                view.setUint32(offset, dim, true);
                offset += 4;
            }

            const floatData = data instanceof Float32Array ? data : new Float32Array(data);

            if (isInt8) {
                // Quantize: find scale, write scale + int8 data
                let absMax = 0;
                for (let i = 0; i < floatData.length; i++) {
                    const abs = Math.abs(floatData[i]);
                    if (abs > absMax) absMax = abs;
                }
                const scale = absMax > 0 ? absMax / 127.0 : 1.0;

                view.setFloat32(offset, scale, true);
                offset += 4;

                const int8Data = new Int8Array(floatData.length);
                for (let i = 0; i < floatData.length; i++) {
                    int8Data[i] = Math.round(Math.max(-127, Math.min(127, floatData[i] / scale)));
                }
                new Uint8Array(buffer, offset, int8Data.length).set(new Uint8Array(int8Data.buffer));
                offset += int8Data.length;

                // Align absolute offset to 4 bytes
                offset = (offset + 3) & ~3;
            } else {
                new Uint8Array(buffer, offset, floatData.byteLength).set(new Uint8Array(floatData.buffer));
                offset += floatData.byteLength;
            }
        }

        return buffer;
    }
}

export { ModelFormat, ARCH_TYPE, QUANT_TYPE, TOKENIZER_TYPE };
