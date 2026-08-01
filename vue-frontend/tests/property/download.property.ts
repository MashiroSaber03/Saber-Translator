import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'

import {
  DOWNLOAD_FORMATS,
  resolveDownloadFileName,
} from '@/composables/useExportImport'

describe('download export contracts', () => {
  const fileName = fc
    .stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_-'), {
      minLength: 1,
      maxLength: 20,
    })
    .map(name => `${name}.png`)

  it('resolves current-image filenames', () => {
    fc.assert(fc.property(
      fileName,
      fc.nat(100),
      fc.constantFrom('translated', 'clean', 'original'),
      (originalFileName, imageIndex, type) => {
        expect(resolveDownloadFileName(originalFileName, imageIndex, type)).toBe(
          `${type}_${originalFileName.replace(/\.[^/.]+$/, '')}.png`,
        )
      },
    ))
  })

  it('uses the current fallback filename and format set', () => {
    expect(resolveDownloadFileName('', 3, 'original')).toBe('original_image_3.png')
    expect(DOWNLOAD_FORMATS).toEqual(['zip', 'pdf', 'cbz'])
  })
})
