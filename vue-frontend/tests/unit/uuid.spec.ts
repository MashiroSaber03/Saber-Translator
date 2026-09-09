import { afterEach, expect, it, vi } from 'vitest'
import { webcrypto } from 'node:crypto'
import { createUuid } from '@/utils/uuid'
import { newIdempotencyKey } from '@/api/v2/content'
import { newProofreadingRoundId } from '@/stores/settings/proofreadingIdentity'

afterEach(() => vi.unstubAllGlobals())
it('generates RFC 4122 v4 UUIDs without the secure-context randomUUID API', () => {
  vi.stubGlobal('crypto', { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) })
  const values = Array.from({ length: 1000 }, createUuid)
  expect(new Set(values).size).toBe(values.length)
  for (const value of [...values, newIdempotencyKey(), newProofreadingRoundId()])
    expect(value).toMatch(/^[\da-f]{8}-[\da-f]{4}-4[\da-f]{3}-[89ab][\da-f]{3}-[\da-f]{12}$/)
})
