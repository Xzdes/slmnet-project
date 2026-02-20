#!/usr/bin/env python3
"""
export_slmnet.py — Convert PyTorch models to .slmnet binary format.

Usage:
    python export_slmnet.py --model model.pt --output model.slmnet --arch transformer
    python export_slmnet.py --model model.pt --output model.slmnet --arch mlp --labels "pos,neg,neu"

Supports:
    - MLP (simple feedforward classifier)
    - Transformer (GPT-like decoder-only)
    - Optional INT8 quantization
"""

import argparse
import json
import struct
import sys

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)


MAGIC = 0x4E4D4C53  # "SLMN" little-endian
HEADER_SIZE = 64

ARCH_MLP = 0
ARCH_TRANSFORMER = 1

QUANT_NONE = 0
QUANT_INT8 = 1

TOK_CHAR = 0
TOK_BPE = 1
TOK_BOW = 2


def write_header(f, header):
    """Write 64-byte header."""
    data = struct.pack('<IIIIIIIIIIII',
        MAGIC,
        header.get('version', 1),
        header.get('arch_type', 0),
        header.get('quantization', 0),
        header.get('vocab_size', 0),
        header.get('embed_dim', 0),
        header.get('num_heads', 0),
        header.get('num_layers', 0),
        header.get('block_size', 0),
        header.get('hidden_dim', 0),
        header.get('tokenizer_type', 0),
        header.get('num_labels', 0),
    )
    data += b'\x00' * (HEADER_SIZE - len(data))  # padding
    f.write(data)


def write_json_section(f, data):
    """Write a JSON section: 4-byte length + JSON bytes."""
    json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
    f.write(struct.pack('<I', len(json_bytes)))
    f.write(json_bytes)


def write_tensor(f, name, tensor, quantize=False):
    """Write a single named tensor."""
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)

    shape = list(tensor.shape)
    f.write(struct.pack('<I', len(shape)))
    for dim in shape:
        f.write(struct.pack('<I', dim))

    data = tensor.detach().cpu().float().numpy()

    if quantize:
        # Simple symmetric INT8 quantization
        abs_max = max(abs(data.min()), abs(data.max()))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        quantized = (data / scale).clip(-127, 127).astype('int8')
        f.write(struct.pack('<f', scale))
        f.write(quantized.tobytes())
        # Align to 4 bytes
        remainder = len(quantized.tobytes()) % 4
        if remainder:
            f.write(b'\x00' * (4 - remainder))
    else:
        f.write(data.tobytes())


def export_transformer(model, output_path, tokenizer_config, tokenizer_type=TOK_CHAR,
                       labels=None, quantize=False, block_size=64):
    """Export a GPT-like transformer model."""
    state = model.state_dict() if hasattr(model, 'state_dict') else model

    # Auto-detect architecture params from state dict
    # Expected keys: token_embed.weight, pos_embed.weight,
    #   block_N.attn.wq.weight, block_N.ln1.gamma, etc.
    token_embed = state.get('token_embed.weight', state.get('token_embedding.weights'))
    vocab_size, embed_dim = token_embed.shape

    # Count layers
    num_layers = 0
    while f'block_{num_layers}.ln1.gamma' in state or f'blocks.{num_layers}.ln1.gamma' in state:
        num_layers += 1

    # Detect num_heads from wq shape
    wq_key = None
    for k in state:
        if 'attn.wq.weight' in k or 'attention.wq.weights' in k:
            wq_key = k
            break

    num_heads = 4  # default
    if wq_key:
        # Heuristic: embed_dim / head_dim, try common values
        for nh in [1, 2, 4, 8, 16]:
            if embed_dim % nh == 0:
                num_heads = nh

    hidden_dim = embed_dim * 4  # default FFN hidden dim
    for k in state:
        if 'ffn.linear1.weight' in k or 'ffn.net.0.weights' in k:
            hidden_dim = state[k].shape[-1]
            break

    header = {
        'version': 1,
        'arch_type': ARCH_TRANSFORMER,
        'quantization': QUANT_INT8 if quantize else QUANT_NONE,
        'vocab_size': int(vocab_size),
        'embed_dim': int(embed_dim),
        'num_heads': int(num_heads),
        'num_layers': int(num_layers),
        'block_size': int(block_size),
        'hidden_dim': int(hidden_dim),
        'tokenizer_type': tokenizer_type,
        'num_labels': len(labels) if labels else 0,
    }

    # Build weight name mapping
    weight_map = {}

    # Try to map from common naming conventions
    for old_key, tensor in state.items():
        new_key = old_key
        # Common renames
        new_key = new_key.replace('token_embedding.weights', 'token_embed.weights')
        new_key = new_key.replace('pos_embedding.weights', 'pos_embed.weights')
        new_key = new_key.replace('.weight', '.weights')
        new_key = new_key.replace('blocks.', 'block_')
        new_key = new_key.replace('.attention.', '.attn.')
        new_key = new_key.replace('.net.0.', '.linear1.')
        new_key = new_key.replace('.net.2.', '.linear2.')
        weight_map[new_key] = tensor

    with open(output_path, 'wb') as f:
        write_header(f, header)
        write_json_section(f, tokenizer_config)
        if labels:
            write_json_section(f, labels)

        # Count and write tensors
        tensor_count = len(weight_map)
        f.write(struct.pack('<I', tensor_count))
        for name, tensor in weight_map.items():
            write_tensor(f, name, tensor, quantize)

    size_kb = round(struct.calcsize('') + 0, 1)
    import os
    file_size = os.path.getsize(output_path)
    print(f"Exported: {output_path}")
    print(f"  Architecture: transformer ({embed_dim}d, {num_heads}h, {num_layers}L)")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Parameters: {sum(t.numel() for t in weight_map.values()):,}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    if quantize:
        print(f"  Quantization: INT8")


def export_mlp(model, output_path, tokenizer_config, tokenizer_type=TOK_BOW,
               labels=None, quantize=False):
    """Export a simple MLP classifier."""
    state = model.state_dict() if hasattr(model, 'state_dict') else model

    # Collect layer tensors
    weight_map = {}
    for key, tensor in state.items():
        new_key = key.replace('.weight', '.weights')
        weight_map[new_key] = tensor

    # Detect input/output dims
    first_weight = list(state.values())[0]
    last_weight = list(state.values())[-2]  # last weight before last bias
    input_dim = first_weight.shape[1] if first_weight.dim() == 2 else first_weight.shape[0]
    output_dim = last_weight.shape[0] if last_weight.dim() == 2 else last_weight.shape[-1]

    header = {
        'version': 1,
        'arch_type': ARCH_MLP,
        'quantization': QUANT_INT8 if quantize else QUANT_NONE,
        'vocab_size': int(input_dim),
        'embed_dim': 0,
        'num_heads': 0,
        'num_layers': 0,
        'block_size': 0,
        'hidden_dim': 0,
        'tokenizer_type': tokenizer_type,
        'num_labels': len(labels) if labels else int(output_dim),
    }

    with open(output_path, 'wb') as f:
        write_header(f, header)
        write_json_section(f, tokenizer_config)
        if labels:
            write_json_section(f, labels)

        f.write(struct.pack('<I', len(weight_map)))
        for name, tensor in weight_map.items():
            write_tensor(f, name, tensor, quantize)

    import os
    file_size = os.path.getsize(output_path)
    print(f"Exported: {output_path}")
    print(f"  Architecture: MLP ({input_dim} -> ... -> {output_dim})")
    print(f"  Parameters: {sum(t.numel() for t in weight_map.values()):,}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to .slmnet format')
    parser.add_argument('--model', required=True, help='Path to PyTorch model (.pt or .pth)')
    parser.add_argument('--output', required=True, help='Output .slmnet file path')
    parser.add_argument('--arch', choices=['mlp', 'transformer'], required=True)
    parser.add_argument('--tokenizer', help='Path to tokenizer config JSON')
    parser.add_argument('--tokenizer-type', choices=['char', 'bpe', 'bow'], default='char')
    parser.add_argument('--labels', help='Comma-separated label names')
    parser.add_argument('--quantize', action='store_true', help='Apply INT8 quantization')
    parser.add_argument('--block-size', type=int, default=64, help='Context window size')

    args = parser.parse_args()

    # Load model
    model = torch.load(args.model, map_location='cpu', weights_only=False)

    # Load tokenizer config
    tok_config = {}
    if args.tokenizer:
        with open(args.tokenizer) as f:
            tok_config = json.load(f)

    tok_type_map = {'char': TOK_CHAR, 'bpe': TOK_BPE, 'bow': TOK_BOW}
    tok_type = tok_type_map[args.tokenizer_type]

    labels = args.labels.split(',') if args.labels else None

    if args.arch == 'transformer':
        export_transformer(model, args.output, tok_config, tok_type, labels,
                          args.quantize, args.block_size)
    else:
        export_mlp(model, args.output, tok_config, tok_type, labels, args.quantize)


if __name__ == '__main__':
    main()
