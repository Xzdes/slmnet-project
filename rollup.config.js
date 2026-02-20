import { nodeResolve } from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';

export default [
    {
        input: 'src/index.js',
        output: {
            file: 'dist/slmnet.min.js',
            format: 'umd',
            name: 'slmnet',
            sourcemap: true,
            exports: 'named',
        },
        plugins: [
            nodeResolve(),
            terser({
                compress: { passes: 2 },
                mangle: { reserved: ['slmnet', 'Model', 'Tensor', 'Ops', 'Sampler'] },
            }),
        ],
    },
    {
        input: 'src/index.js',
        output: {
            file: 'dist/slmnet.js',
            format: 'umd',
            name: 'slmnet',
            sourcemap: true,
            exports: 'named',
        },
        plugins: [nodeResolve()],
    },
];
