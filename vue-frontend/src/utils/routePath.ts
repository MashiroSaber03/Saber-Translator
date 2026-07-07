import { FRONTEND_ROUTE_PATHS } from '@/constants/routes'

export interface RouteClassification {
  isFrontendRoute: boolean
  isApiRoute: boolean
  isStaticRoute: boolean
}

const STATIC_ROUTE_PREFIXES = ['/static/', '/js/', '/assets/'] as const

function stripLeadingSlashes(value: string): string {
  return value.replace(/^\/+/, '')
}

function stripBoundarySlashes(value: string): string {
  return value.replace(/^\/+|\/+$/g, '')
}

function pathnameOf(path: string): string {
  return normalizeAppPath(path).split(/[?#]/)[0] ?? '/'
}

export function normalizeAppPath(path: string): string {
  const trimmedPath = path.trim()

  if (!trimmedPath) {
    return '/'
  }

  return trimmedPath.startsWith('/') ? trimmedPath : `/${trimmedPath}`
}

export function isKnownFrontendRoute(path: string): boolean {
  const pathname = pathnameOf(path)
  return FRONTEND_ROUTE_PATHS.includes(pathname as (typeof FRONTEND_ROUTE_PATHS)[number])
}

export function classifyAppPath(path: string): RouteClassification {
  const pathname = pathnameOf(path)
  const isApiRoute = pathname === '/api' || pathname.startsWith('/api/')
  const isStaticRoute = pathname === '/static' || STATIC_ROUTE_PREFIXES.some(prefix => pathname.startsWith(prefix))

  return {
    isFrontendRoute: !isApiRoute && !isStaticRoute,
    isApiRoute,
    isStaticRoute,
  }
}

export function buildApiPath(endpoint: string): string {
  const cleanEndpoint = stripBoundarySlashes(endpoint)
  return cleanEndpoint ? `/api/${cleanEndpoint}` : '/api'
}

export function buildStaticPath(resourceType: string, filename: string): string {
  const cleanResourceType = stripBoundarySlashes(resourceType)
  const cleanFilename = stripLeadingSlashes(filename)
  return `/static/${cleanResourceType}/${cleanFilename}`
}

export function buildVueStaticAssetPath(filename: string): string {
  const cleanFilename = stripLeadingSlashes(filename).replace(/^(assets|js)\//, '')
  return cleanFilename.endsWith('.js') ? `/js/${cleanFilename}` : `/assets/${cleanFilename}`
}
