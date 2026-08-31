import { createHash } from 'node:crypto'
import {
  copyFile,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { execFileSync } from 'node:child_process'

const root = resolve(import.meta.dirname, '..')
const dist = resolve(root, 'dist')
const release = resolve(root, 'release')
const packageMetadata = JSON.parse(
  await readFile(resolve(root, 'package.json'), 'utf8'),
)
const version = packageMetadata.version

if (typeof version !== 'string' || !/^\d+(?:\.\d+){2,3}$/.test(version)) {
  throw new Error('package.json contains an invalid extension version')
}

const artifactNames = {
  local: `saber-translator-browser-extension-v${version}.zip`,
  chrome: `saber-translator-chrome-web-store-v${version}.zip`,
  edge: `saber-translator-edge-addons-v${version}.zip`,
}

function powershellLiteral(value) {
  return `'${value.replaceAll("'", "''")}'`
}

function compressDirectory(source, destination) {
  execFileSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-Command',
      `Compress-Archive -Path ${powershellLiteral(join(source, '*'))} -DestinationPath ${powershellLiteral(destination)} -Force`,
    ],
    { stdio: 'inherit' },
  )
}

async function writeChecksum(archive) {
  const digest = createHash('sha256').update(await readFile(archive)).digest('hex')
  await writeFile(
    `${archive}.sha256`,
    `${digest}  ${basename(archive)}\n`,
    'utf8',
  )
}

await rm(release, { recursive: true, force: true })
await mkdir(release, { recursive: true })

const localArchive = resolve(release, artifactNames.local)
compressDirectory(dist, localArchive)

const storeStaging = await mkdtemp(join(tmpdir(), 'saber-extension-store-'))
try {
  await cp(dist, storeStaging, { recursive: true })
  const storeManifestPath = join(storeStaging, 'manifest.json')
  const storeManifest = JSON.parse(await readFile(storeManifestPath, 'utf8'))
  if (
    storeManifest.manifest_version !== 3
    || storeManifest.version !== version
    || typeof storeManifest.key !== 'string'
  ) {
    throw new Error('built manifest does not match package.json or has no development key')
  }
  delete storeManifest.key
  await writeFile(
    storeManifestPath,
    `${JSON.stringify(storeManifest, null, 2)}\n`,
    'utf8',
  )
  await rm(join(storeStaging, 'README.md'), { force: true })

  const chromeArchive = resolve(release, artifactNames.chrome)
  const edgeArchive = resolve(release, artifactNames.edge)
  compressDirectory(storeStaging, chromeArchive)
  await copyFile(chromeArchive, edgeArchive)

  await Promise.all(
    [localArchive, chromeArchive, edgeArchive].map(writeChecksum),
  )
} finally {
  await rm(storeStaging, { recursive: true, force: true })
}
