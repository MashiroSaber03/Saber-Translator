import { apiClient } from '@/api/client'
import type {
  V2CredentialEdit,
  V2CredentialSummary,
  V2SettingsTransaction,
} from '@/api/v2/settings'

const DATABASE_NAME = 'saber-public-credentials'
const STORE_NAME = 'credentials'

interface BrowserCredentialRecord {
  key: string
  ownerUserId: string
  domain: string
  provider: string
  secret: Record<string, string>
}

let enabled = false
let ownerUserId = ''

export function configureBrowserCredentials(active: boolean, userId = ''): void {
  enabled = active
  ownerUserId = active ? userId : ''
}

export function usesBrowserCredentials(): boolean {
  return enabled && Boolean(ownerUserId)
}

function recordKey(domain: string, provider: string): string {
  return `${ownerUserId}\0${domain}\0${provider}`
}

function openDatabase(): Promise<IDBDatabase> {
  if (!globalThis.indexedDB) {
    return Promise.reject(new Error('当前浏览器不支持本地密钥存储'))
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1)
    request.onupgradeneeded = () => {
      const database = request.result
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: 'key' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('无法打开浏览器密钥库'))
  })
}

async function transact<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const database = await openDatabase()
  try {
    return await new Promise<T>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, mode)
      const request = operation(transaction.objectStore(STORE_NAME))
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error ?? new Error('浏览器密钥库操作失败'))
      transaction.onabort = () => reject(transaction.error ?? new Error('浏览器密钥库事务失败'))
    })
  } finally {
    database.close()
  }
}

function validatedSecret(value: Record<string, unknown>): Record<string, string> {
  const entries = Object.entries(value)
  if (entries.length === 0 || entries.some(([, child]) => typeof child !== 'string')) {
    throw new Error('API Key 内容无效')
  }
  return Object.fromEntries(entries) as Record<string, string>
}

async function loadRecords(): Promise<BrowserCredentialRecord[]> {
  if (!usesBrowserCredentials()) return []
  const records = await transact<BrowserCredentialRecord[]>('readonly', store => store.getAll())
  return records.filter(record => record.ownerUserId === ownerUserId)
}

async function uploadLease(record: BrowserCredentialRecord): Promise<void> {
  await apiClient.put(
    `/api/v2/browser-credentials/${encodeURIComponent(record.domain)}/${encodeURIComponent(record.provider)}`,
    { secret: record.secret },
  )
}

function summary(record: BrowserCredentialRecord): V2CredentialSummary {
  const reference = `browser:${record.domain}:${record.provider}`
  return {
    credentialId: reference,
    credentialVersionId: reference,
    domain: record.domain,
    provider: record.provider,
    hasKey: true,
    currentVersion: 1,
    revision: 0,
  }
}

export async function restoreBrowserCredentialLeases(): Promise<V2CredentialSummary[]> {
  const records = await loadRecords()
  await Promise.all(records.map(uploadLease))
  return records.map(summary)
}

async function saveCredentialEdit(edit: V2CredentialEdit): Promise<BrowserCredentialRecord> {
  const record: BrowserCredentialRecord = {
    key: recordKey(edit.domain, edit.provider),
    ownerUserId,
    domain: edit.domain,
    provider: edit.provider,
    secret: validatedSecret(edit.secret),
  }
  await transact<IDBValidKey>('readwrite', store => store.put(record))
  await uploadLease(record)
  return record
}

export async function prepareBrowserCredentialTransaction(
  transaction: V2SettingsTransaction,
): Promise<{
  transaction: V2SettingsTransaction
  summaries: V2CredentialSummary[]
}> {
  if (!usesBrowserCredentials()) return { transaction, summaries: [] }
  const records = await Promise.all(
    (transaction.credentialEdits ?? []).map(saveCredentialEdit),
  )
  return {
    transaction: {
      ...transaction,
      providerSettings: (transaction.providerSettings ?? []).map(row => {
        const clean = { ...row }
        delete clean.credentialEditRef
        delete clean.credentialVersionId
        return clean
      }),
      credentialEdits: [],
    },
    summaries: records.map(summary),
  }
}
