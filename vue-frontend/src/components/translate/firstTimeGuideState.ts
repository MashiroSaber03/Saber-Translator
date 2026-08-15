export const DISMISS_SETUP_REMINDER_KEY = 'saber_translator_dismiss_setup_reminder'

function resolveStorage(storage?: Storage): Storage | null {
  if (storage) {
    return storage
  }

  try {
    return window.localStorage
  } catch {
    return null
  }
}

export function shouldShowFirstTimeGuide(storage?: Storage): boolean {
  try {
    return resolveStorage(storage)?.getItem(DISMISS_SETUP_REMINDER_KEY) !== 'true'
  } catch {
    return true
  }
}

export function dismissFirstTimeGuide(storage?: Storage): void {
  try {
    resolveStorage(storage)?.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
  } catch {
    // The mounted guide can still close even when persistence is unavailable.
  }
}
