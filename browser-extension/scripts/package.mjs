import { createHash } from 'node:crypto'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { execFileSync } from 'node:child_process'

const root = resolve(import.meta.dirname, '..')
const release = resolve(root, 'release')
const archive = resolve(release, 'saber-translator-browser-extension-v1.0.0.zip')

await rm(release, { recursive: true, force: true })
await mkdir(release, { recursive: true })
execFileSync(
  'powershell.exe',
  [
    '-NoProfile',
    '-Command',
    `Compress-Archive -Path '${resolve(root, 'dist', '*')}' -DestinationPath '${archive}' -Force`,
  ],
  { stdio: 'inherit' },
)
const digest = createHash('sha256').update(await readFile(archive)).digest('hex')
await writeFile(`${archive}.sha256`, `${digest}  ${archive.split(/[\\/]/).at(-1)}\n`, 'utf8')
