import { afterEach, describe, expect, it, vi } from 'vitest'
import { triggerBlobDownload, triggerUrlDownload } from '@/utils/browserDownload'

const originalCreateObjectURL = URL.createObjectURL
const originalRevokeObjectURL = URL.revokeObjectURL

describe('browserDownload utilities', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: originalCreateObjectURL,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: originalRevokeObjectURL,
    })
  })

  it('downloads an existing URL through a temporary anchor', () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const appendSpy = vi.spyOn(document.body, 'appendChild')

    triggerUrlDownload('/api/download/file')

    const anchor = appendSpy.mock.calls[0]?.[0] as HTMLAnchorElement
    expect(anchor).toBeInstanceOf(HTMLAnchorElement)
    expect(anchor.href).toContain('/api/download/file')
    expect(anchor.download).toBe('')
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(anchor.isConnected).toBe(false)
  })

  it('revokes object URLs even when a blob download click fails', () => {
    const createObjectURL = vi.fn(() => 'blob:download')
    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: createObjectURL,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: revokeObjectURL,
    })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
      throw new Error('click failed')
    })

    expect(() => {
      triggerBlobDownload(new Blob(['content'], { type: 'text/plain' }), 'notes.txt')
    }).toThrow('click failed')

    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:download')
  })
})
