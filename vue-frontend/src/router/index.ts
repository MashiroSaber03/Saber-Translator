/**
 * Vue Router 配置
 * 配置应用的路由系统，支持书架、翻译、阅读器和漫画分析四个主要页面
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

// 路由配置
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'bookshelf',
    component: () => import('@/views/BookshelfView.vue'),
    meta: { title: '书架' }
  },
  {
    path: '/translate',
    name: 'translate',
    component: () => import('@/views/TranslateView.vue'),
    meta: { title: '翻译' },
    // 支持可选的 book 和 chapter 查询参数
    props: (route) => ({
      bookId: route.query.book as string | undefined,
      chapterId: route.query.chapter as string | undefined
    })
  },
  {
    path: '/reader',
    name: 'reader',
    component: () => import('@/views/ReaderView.vue'),
    meta: { title: '阅读器' },
    // 要求 book 和 chapter 参数
    props: (route) => ({
      bookId: route.query.book as string,
      chapterId: route.query.chapter as string
    }),
    beforeEnter: (to, _from, next) => {
      // 验证必需参数
      if (!to.query.book || !to.query.chapter) {
        // 缺少参数时重定向到书架
        next({ name: 'bookshelf' })
      } else {
        next()
      }
    }
  },
  {
    path: '/insight',
    name: 'insight',
    component: () => import('@/views/InsightView.vue'),
    meta: { title: '漫画分析' },
    props: (route) => ({
      bookId: route.query.book as string | undefined
    })
  },
  {
    // 未定义路由重定向到书架
    path: '/:pathMatch(.*)*',
    redirect: { name: 'bookshelf' }
  }
]

// 创建路由实例
const router = createRouter({
  history: createWebHistory('/'),
  routes
})

// 路由守卫 - 更新页面标题
router.beforeEach((to, _from, next) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} - Saber Translator` : 'Saber Translator'
  next()
})

export default router
