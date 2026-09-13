import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const manifest = JSON.parse(
  readFileSync(new URL('../public/manifest.json', import.meta.url), 'utf8'),
) as {
  manifest_version: number
  name: string
  description: string
  default_locale: string
  action: { default_title: string }
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
  it('declares Chinese store metadata with resolvable locale messages', () => {
    expect(manifest.default_locale).toBe('zh_CN')
    const messages = JSON.parse(readFileSync(
      new URL(`../public/_locales/${manifest.default_locale}/messages.json`, import.meta.url),
      'utf8',
    ))
    for (const value of [manifest.name, manifest.description, manifest.action.default_title]) {
      expect(value).toMatch(/^__MSG_\w+__$/)
      const key = value.slice(6, -2)
      expect(messages[key]?.message).toEqual(expect.stringMatching(/[\u4e00-\u9fff]/))
    }
  })

  it('uses the package version as the release version', () => {
    expect(manifest.version).toBe(packageMetadata.version)
  })

  it('keeps the fixed MV3 identity and only the required named permissions', () => {
    expect(manifest.manifest_version).toBe(3)
    expect(manifest.key).toMatch(/^MIIB/)
    expect(manifest.permissions).toEqual(['storage', 'contextMenus', 'alarms'])
    expect(manifest).not.toHaveProperty('side_panel')
    expect(manifest.action).not.toHaveProperty('default_popup')
    expect(manifest).toHaveProperty('web_accessible_resources', [{ resources: ['panel.html', 'assets/*'], matches: ['http://*/*', 'https://*/*'] }])
    expect(manifest.host_permissions).toEqual(['http://*/*', 'https://*/*'])
    expect(manifest.content_scripts[0]?.matches).toEqual([
      'http://*/*',
      'https://*/*',
    ])
  })
})
