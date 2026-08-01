export class BackendAccessRestrictedError extends Error {
  readonly code = 'SETTINGS_RESTRICTED'
  readonly status = 503

  constructor(message: string) {
    super(message)
    this.name = 'BackendAccessRestrictedError'
    Object.setPrototypeOf(this, new.target.prototype)
  }
}

let restricted = false
let restrictedReason = '后端设置尚未加载，当前处于受限模式'

export function setBackendAccessRestricted(
  value: boolean,
  reason?: string | null,
): void {
  restricted = value
  if (reason?.trim()) restrictedReason = reason.trim()
}

export function assertBackendActionAllowed(): void {
  if (!restricted) return
  throw new BackendAccessRestrictedError(
    `${restrictedReason}；已阻止保存、创建任务和 Provider 调用`,
  )
}
