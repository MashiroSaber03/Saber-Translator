export const UPDATE_KEY = 'saber-pending-store-update'

export async function isStoreInstallation(): Promise<boolean> {
  const info = await chrome.management.getSelf()
  if (info.installType !== 'normal' || !info.updateUrl) return false
  const url = new URL(info.updateUrl)
  return url.protocol === 'https:' && (
    (url.hostname === 'clients2.google.com' && url.pathname === '/service/update2/crx')
    || (url.hostname === 'edge.microsoft.com' && url.pathname === '/extensionwebstorebase/v1/crx')
  )
}

export function registerUpdateNotifications(): void {
  chrome.runtime.onUpdateAvailable.addListener(details => {
    void isStoreInstallation().then(store => {
      if (store) return chrome.storage.session.set({ [UPDATE_KEY]: details.version })
    }).catch(() => undefined)
  })
  chrome.runtime.onInstalled.addListener(details => {
    if (details.reason === 'update') void chrome.storage.session.remove(UPDATE_KEY)
  })
}
