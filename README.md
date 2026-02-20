# slmnet

Lightweight zero-dependency JavaScript runtime for neural network inference in the browser.

## Features

- **Zero dependencies** — single `<script>` tag, no npm required
- **Browser-native** — runs entirely client-side, no server needed
- **MLP + Transformer** — supports classification and text generation architectures
- **Binary model format** — compact `.slmnet` format with optional int8 quantization
- **Pipeline engine** — chain tokenizers, models, and decoders into a single `run()` call
- **Tiny footprint** — minified bundle under 30KB

## Quick Start

```html
<script src="https://cdn.jsdelivr.net/npm/slmnet@1/dist/slmnet.min.js"></script>
<script>
  (async () => {
    const model = await slmnet.Model.load('/my-model.slmnet');

    // Classification
    const { label, score } = model.classify('I love this product');
    console.log(label, score); // "positive" 0.92

    // Text generation
    const text = model.generate('Once upon', { maxTokens: 50 });
    console.log(text);
  })();
</script>
```

Or via ES modules:

```js
import slmnet from 'slmnet';

const result = await slmnet.run('./pipeline.json', 'input text');
```

## Installation

**CDN (recommended for browsers):**

```html
<script src="https://cdn.jsdelivr.net/npm/slmnet@1/dist/slmnet.min.js"></script>
```

**npm:**

```bash
npm install slmnet
```

## API Reference

### `slmnet.Model`

Load and run neural network models.

```js
// Load a model (with optional progress callback)
const model = await slmnet.Model.load('/model.slmnet', {
  onProgress: (p) => console.log(`${(p * 100).toFixed(0)}%`)
});

// Classification (MLP architecture)
const { label, score, scores } = model.classify('some text');
console.log(label);  // "positive"
console.log(score);  // 0.87
console.log(scores); // { positive: 0.87, negative: 0.13 }

// Text generation (Transformer architecture)
const text = model.generate('prompt', {
  maxTokens: 100,
  temperature: 0.8,
  topK: 40,
  topP: 0.9,
  onToken: (token) => document.body.append(token), // streaming
  shouldStop: (text) => text.includes('.'),          // early stop
});
```

### `slmnet.run()`

Execute a processing pipeline — chains tokenizers, models, logic gates, and decoders.

```js
const ctx = await slmnet.run('./pipeline.json', 'user input', {
  onProgress: (fraction) => console.log(`${(fraction * 100).toFixed(0)}%`),
  onStepStart: ({ step, index, total }) => console.log(`Step ${index + 1}/${total}`),
  onStepComplete: ({ step, result }) => console.log(result),
});
```

Pipeline config example:

```json
{
  "input_field": "text",
  "pipeline": [
    { "id": "tokens", "type": "tokenizer", "input": "text", "vocabulary": ["hello", "world"] },
    { "id": "prediction", "type": "neural_model", "input": "tokens", "model": "/model.slmnet" },
    { "id": "label", "type": "decoder", "input": "prediction", "categories": ["positive", "negative"] }
  ]
}
```

### `slmnet.Tensor`

N-dimensional typed array container.

```js
const t = new slmnet.Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const zeros = slmnet.Tensor.zeros([4, 4]);
const ones = slmnet.Tensor.ones([3]);
const reshaped = t.reshape([3, 2]);
```

### `slmnet.Ops`

Pure math operations on tensors.

```js
const { Ops, Tensor } = slmnet;

const a = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
const b = new Tensor(new Float32Array([5, 6, 7, 8]), [2, 2]);

const c = Ops.matMul(a, b);       // Matrix multiplication
const d = Ops.add(a, b);          // Element-wise addition
const e = Ops.relu(a);            // ReLU activation
const f = Ops.softmax(a);         // Softmax (last dim)
const g = Ops.layerNorm(a, gamma, beta); // Layer normalization
```

### `slmnet.Sampler`

Token sampling with temperature, top-k, and top-p.

```js
const logits = new slmnet.Tensor(new Float32Array([1.0, 5.0, 2.0]), [3]);
const tokenId = slmnet.Sampler.sample(logits, {
  temperature: 0.8,
  topK: 10,
  topP: 0.9,
});
```

### `slmnet.ModelFormat`

Parse and build the binary `.slmnet` model format.

```js
// Parse
const buffer = await fetch('/model.slmnet').then(r => r.arrayBuffer());
const { header, tokenizerConfig, labels, weights } = slmnet.ModelFormat.parse(buffer);

// Build
const outputBuffer = slmnet.ModelFormat.build({ header, tokenizerConfig, labels, weights });
```

### `slmnet.Loader`

Resource loader with caching and timeout.

```js
const json = await slmnet.Loader.loadJson('/config.json');
const buffer = await slmnet.Loader.loadBinary('/model.slmnet');

// Custom options
const data = await slmnet.Loader.loadBinary('/large-model.slmnet', { timeout: 60000 });
```

### Error Classes

All errors extend `SlmnetError` for unified `catch` handling:

```js
try {
  await model.classify('test');
} catch (err) {
  if (err instanceof slmnet.ShapeError) {
    console.error('Tensor shape mismatch:', err.details);
  } else if (err instanceof slmnet.ModelLoadError) {
    console.error('Failed to load:', err.details.url, err.details.status);
  } else if (err instanceof slmnet.PipelineError) {
    console.error('Pipeline step failed:', err.details.stepId);
  } else if (err instanceof slmnet.SlmnetError) {
    console.error('slmnet error:', err.message);
  }
}
```

### Constants

```js
slmnet.ARCH_TYPE       // { MLP: 0, TRANSFORMER: 1 }
slmnet.QUANT_TYPE      // { FLOAT32: 0, INT8: 1 }
slmnet.TOKENIZER_TYPE  // { CHAR: 0, BPE: 1, BOW: 2 }
slmnet.version         // "1.0.0"
```

## Model Format

The `.slmnet` binary format stores everything needed for inference in a single file:

| Section | Contents |
|---------|----------|
| Header (64 bytes) | Magic, version, architecture, quantization, dimensions |
| Tokenizer | JSON-encoded vocabulary and config |
| Labels | JSON-encoded label array (for classifiers) |
| Weights | Named tensors with shape metadata |

**Quantization:** Set `quantization: 1` (INT8) in the header to store weights as 8-bit integers with per-tensor scaling. Reduces model size ~4x with minimal accuracy loss.

Use `tools/generate_sample_weights.js` to generate sample model weights for testing.

## Pipeline Handlers

| Type | Description |
|------|-------------|
| `tokenizer` | Text to token vector (BoW, char, BPE) |
| `neural_model` | Run inference on a `.slmnet` model |
| `decoder` | Map prediction vector to category label |
| `vector_builder` | Build one-hot encoded feature vectors |
| `logic_gate` | Rule-based routing with threshold conditions |
| `generator` | Text generation with sampling |

## Development

```bash
# Install dependencies
npm install

# Build UMD bundles (dist/slmnet.js + dist/slmnet.min.js)
npm run build

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Lint
npm run lint

# Format code
npm run format
```

## Project Structure

```
src/
  index.js              # Public API entry point
  errors.js             # Error class hierarchy
  Loader.js             # Fetch + cache
  core/
    Tensor.js           # N-dim typed array
    Ops.js              # Math operations
  nn/
    Linear.js           # Dense layer
    Embedding.js        # Token embeddings
    LayerNorm.js        # Layer normalization
    Attention.js        # Multi-head self-attention
    FeedForward.js      # FFN block
    TransformerBlock.js # Attention + FFN + residual
    MLP.js              # Multi-layer perceptron
  runtime/
    Model.js            # High-level model API
    ModelFormat.js       # Binary format parser/builder
    Sampler.js          # Token sampling strategies
  tokenizer/
    CharTokenizer.js    # Character-level tokenizer
    BPETokenizer.js     # Byte-pair encoding tokenizer
  pipeline/
    Executor.js         # Pipeline runner
    handlers/           # Step handlers (tokenizer, neural, decoder, etc.)
```

## License

MIT
