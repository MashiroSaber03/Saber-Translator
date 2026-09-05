import { build } from 'vite'
import { copyFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const dist = resolve(root, 'dist')

async function buildIife(entry, name, emptyOutDir) {
  await build({
    root,
    publicDir: false,
    build: {
      target: 'chrome120',
      outDir: dist,
      emptyOutDir,
      lib: {
        entry: resolve(root, `src/${entry}.ts`),
        name,
        formats: ['iife'],
        fileName: () => `${entry}.js`,
      },
      rollupOptions: {
        output: { inlineDynamicImports: true },
      },
    },
  })
}

await buildIife('content', 'SaberTranslatorContent', true)
await buildIife('background', 'SaberTranslatorBackground', false)
await build({
  root,
  publicDir: resolve(root, 'public'),
  build: {
    target: 'chrome120',
    outDir: dist,
    emptyOutDir: false,
    rollupOptions: {
      input: resolve(root, 'popup.html'),
    },
  },
})
await Promise.all([
  copyFile(resolve(root, 'README.md'), resolve(dist, 'README.md')),
  copyFile(resolve(root, 'PRIVACY.md'), resolve(dist, 'PRIVACY.md')),
  copyFile(resolve(root, '..', 'LICENSE'), resolve(dist, 'LICENSE')),
  copyFile(
    resolve(root, 'THIRD_PARTY_NOTICES.md'),
    resolve(dist, 'THIRD_PARTY_NOTICES.md'),
  ),
])
