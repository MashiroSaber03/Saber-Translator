import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const manifest = JSON.parse(
  readFileSync(new URL('../public/manifest.json', import.meta.url), 'utf8'),
) as {
  manifest_version: number
  version: string
  key: string
  permissions: string[]
  host_permissions: string[]
  content_scripts: Array<{ matches: string[] }>
}
const packageMetadata = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
) as { version: string }

describe('extension manifest', () => {
  it('uses the package version as the release version', () => {
    expect(manifest.version).toBe(packageMetadata.version)
  })

  it('keeps the fixed MV3 identity and only the required named permissions', () => {
    expect(manifest.manifest_version).toBe(3)
    expect(manifest.key).toMatch(/^MIIB/)
    expect(manifest.permissions).toEqual(['storage', 'contextMenus'])
    expect(manifest.host_permissions).toEqual(['http://*/*', 'https://*/*'])
    expect(manifest.content_scripts[0]?.matches).toEqual([
      'http://*/*',
      'https://*/*',
    ])
  })
})
