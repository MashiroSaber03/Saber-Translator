/**
 * 路由路径解析属性测试
 * 使用 fast-check 进行属性基测试，验证路由路径解析的一致性
 * 
 * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
 * Validates: Requirements 11.2, 11.3
 */
import { describe, it } from 'vitest'
import * as fc from 'fast-check'

/**
 * 路由路径分类器
 * 用于判断路径应该由前端路由还是后端 API 处理
 */
interface RouteClassification {
  isFrontendRoute: boolean  // 是否为前端路由
  isApiRoute: boolean       // 是否为 API 路由
  isStaticRoute: boolean    // 是否为静态资源路由
}

/**
 * 分类路由路径
 * @param path 路由路径
 * @returns 路由分类结果
 */
function classifyRoute(path: string): RouteClassification {
  // 规范化路径
  const normalizedPath = path.startsWith('/') ? path : '/' + path

  // API 路由：以 /api/ 开头
  const isApiRoute = normalizedPath.startsWith('/api/')

  // 静态资源路由：以 /static/ 开头
  const isStaticRoute = normalizedPath.startsWith('/static/')

  // 前端路由：非 API 且非静态资源
  const isFrontendRoute = !isApiRoute && !isStaticRoute

  return {
    isFrontendRoute,
    isApiRoute,
    isStaticRoute,
  }
}

/**
 * 判断路径是否为有效的前端路由
 * @param path 路由路径
 * @returns 是否为有效的前端路由
 */
function isValidFrontendRoute(path: string): boolean {
  const validRoutes = ['/', '/translate', '/reader', '/insight']
  const normalizedPath = path.startsWith('/') ? path : '/' + path

  // 检查是否为有效的前端路由（包括带查询参数的情况）
  const pathWithoutQuery = normalizedPath.split('?')[0] ?? ''
  return validRoutes.includes(pathWithoutQuery)
}

/**
 * 构建 API 路径
 * @param endpoint API 端点名称
 * @returns 完整的 API 路径
 */
function buildApiPath(endpoint: string): string {
  // 确保端点不以斜杠开头
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint
  return `/api/${cleanEndpoint}`
}

/**
 * 构建静态资源路径
 * @param resourceType 资源类型（css、js、fonts、pic）
 * @param filename 文件名
 * @returns 完整的静态资源路径
 */
function buildStaticPath(resourceType: string, filename: string): string {
  return `/static/${resourceType}/${filename}`
}

/**
 * 构建 Vue 静态资源路径
 * Vue 静态资源现在使用根路径 /js/ 和 /assets/
 * @param filename 文件名
 * @returns 完整的 Vue 静态资源路径
 */
function buildVueStaticPath(filename: string): string {
  // 根据文件类型返回正确的路径
  if (filename.endsWith('.js')) {
    return `/js/${filename}`
  } else if (filename.endsWith('.css') || filename.includes('assets/')) {
    // 移除 assets/ 前缀如果存在
    const cleanFilename = filename.replace('assets/', '')
    return `/assets/${cleanFilename}`
  }
  // 默认返回 assets 路径
  return `/assets/${filename}`
}

describe('路由路径解析属性测试', () => {
  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * 对于任意 API 路径，应该被正确分类为 API 路由
   */
  it('API 路径应该被正确分类为 API 路由', () => {
    // 定义常见的 API 端点
    const apiEndpoints = [
      'bookshelf/books',
      'sessions/save',
      'plugins',
      'get_prompts',
      'manga-insight/test/analyze/start',
      'test_ollama_connection',
    ]

    fc.assert(
      fc.property(
        fc.constantFrom(...apiEndpoints),
        (endpoint) => {
          const path = buildApiPath(endpoint)
          const classification = classifyRoute(path)

          // API 路径应该被分类为 API 路由
          return classification.isApiRoute === true &&
            classification.isFrontendRoute === false &&
            classification.isStaticRoute === false
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * 对于任意静态资源路径，应该被正确分类为静态资源路由
   */
  it('静态资源路径应该被正确分类为静态资源路由', () => {
    // 定义资源类型和示例文件名
    const resourceTypes = ['css', 'js', 'fonts', 'pic', 'vue']
    const filenames = ['style.css', 'main.js', 'font.woff2', 'logo.png', 'index.html']

    fc.assert(
      fc.property(
        fc.constantFrom(...resourceTypes),
        fc.constantFrom(...filenames),
        (resourceType, filename) => {
          const path = buildStaticPath(resourceType, filename)
          const classification = classifyRoute(path)

          // 静态资源路径应该被分类为静态资源路由
          return classification.isStaticRoute === true &&
            classification.isFrontendRoute === false &&
            classification.isApiRoute === false
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * 对于任意前端路由路径，应该被正确分类为前端路由
   */
  it('前端路由路径应该被正确分类为前端路由', () => {
    const frontendRoutes = ['/', '/translate', '/reader', '/insight']

    fc.assert(
      fc.property(
        fc.constantFrom(...frontendRoutes),
        (path) => {
          const classification = classifyRoute(path)

          // 前端路由路径应该被分类为前端路由
          return classification.isFrontendRoute === true &&
            classification.isApiRoute === false &&
            classification.isStaticRoute === false
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * 路由分类应该是互斥的（一个路径只能属于一种类型）
   */
  it('路由分类应该是互斥的', () => {
    // 生成各种类型的路径
    const allPaths = [
      '/',
      '/translate',
      '/reader',
      '/insight',
      '/api/bookshelf/books',
      '/static/css/style.css',
      '/js/main.abc123.js',     // Vue JS 资源
      '/assets/index.abc123.css', // Vue CSS 资源
      '/unknown/path',
    ]

    fc.assert(
      fc.property(
        fc.constantFrom(...allPaths),
        (path) => {
          const classification = classifyRoute(path)

          // 计算为 true 的分类数量
          const trueCount = [
            classification.isApiRoute,
            classification.isStaticRoute,
          ].filter(Boolean).length

          // API 和静态资源是互斥的
          // 前端路由是"其他"类型，所以当 API 或静态资源为 true 时，前端路由应该为 false
          if (classification.isApiRoute || classification.isStaticRoute) {
            return classification.isFrontendRoute === false && trueCount === 1
          }

          // 如果既不是 API 也不是静态资源，则应该是前端路由
          return classification.isFrontendRoute === true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * Vue 静态资源路径应该正确构建（使用新的 /js/ 和 /assets/ 路径）
   */
  it('Vue 静态资源路径应该正确构建', () => {
    // JS 文件
    const jsFiles = ['main.abc123.js', 'vue-vendor.def456.js']
    // 资源文件（CSS、图片等）
    const assetFiles = ['index.abc123.css', 'logo.ghi789.png']

    fc.assert(
      fc.property(
        fc.constantFrom(...jsFiles),
        (filename) => {
          const path = buildVueStaticPath(filename)
          // 验证 JS 文件路径格式正确
          return path.startsWith('/js/') && path.endsWith(filename)
        }
      ),
      { numRuns: 100 }
    )

    fc.assert(
      fc.property(
        fc.constantFrom(...assetFiles),
        (filename) => {
          const path = buildVueStaticPath(filename)
          // 验证资源文件路径格式正确
          return path.startsWith('/assets/') && path.endsWith(filename)
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * 带查询参数的前端路由应该被正确识别
   */
  it('带查询参数的前端路由应该被正确识别', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('/translate', '/reader', '/insight'),
        fc.record({
          book: fc.option(fc.hexaString({ minLength: 8, maxLength: 8 }), { nil: undefined }),
          chapter: fc.option(fc.hexaString({ minLength: 8, maxLength: 8 }), { nil: undefined }),
        }),
        (basePath, params) => {
          // 构建带查询参数的路径
          const queryParts: string[] = []
          if (params.book) queryParts.push(`book=${params.book}`)
          if (params.chapter) queryParts.push(`chapter=${params.chapter}`)

          const fullPath = queryParts.length > 0
            ? `${basePath}?${queryParts.join('&')}`
            : basePath

          // 验证路径被正确识别为前端路由
          return isValidFrontendRoute(fullPath)
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Feature: vue-frontend-migration, Property 45: 路由路径解析一致性
   * Validates: Requirements 11.2, 11.3
   * 
   * API 路径构建应该保持一致性
   */
  it('API 路径构建应该保持一致性', () => {
    fc.assert(
      fc.property(
        // 生成随机的 API 端点名称（只包含字母、数字、下划线和斜杠）
        fc.stringOf(
          fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_/'.split('')),
          { minLength: 1, maxLength: 30 }
        ).filter(s => !s.startsWith('/') && !s.endsWith('/') && !s.includes('//')),
        (endpoint) => {
          const path1 = buildApiPath(endpoint)
          const path2 = buildApiPath('/' + endpoint)

          // 无论端点是否以斜杠开头，构建的路径应该相同
          return path1 === path2
        }
      ),
      { numRuns: 100 }
    )
  })
})

// 导出工具函数供其他测试使用
export {
  classifyRoute,
  isValidFrontendRoute,
  buildApiPath,
  buildStaticPath,
  buildVueStaticPath,
}
export type { RouteClassification }
