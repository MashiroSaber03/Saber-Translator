export const APP_ROUTE_PATHS = {
  bookshelf: '/',
  translate: '/translate',
  reader: '/reader',
  insight: '/insight',
  characterStudio: '/insight/character-studio',
} as const

export const FRONTEND_ROUTE_PATHS = [
  APP_ROUTE_PATHS.bookshelf,
  APP_ROUTE_PATHS.translate,
  APP_ROUTE_PATHS.reader,
  APP_ROUTE_PATHS.insight,
  APP_ROUTE_PATHS.characterStudio,
] as const

export type AppRoutePath = typeof FRONTEND_ROUTE_PATHS[number]
