const PROOFREADING_ROUND_ID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

export function isProofreadingRoundId(value: unknown): value is string {
  return typeof value === 'string' && PROOFREADING_ROUND_ID_PATTERN.test(value)
}

export function newProofreadingRoundId(): string {
  return crypto.randomUUID()
}

export function proofreadingProviderDomain(roundId: string): string {
  return `proofreading_${roundId}`
}
