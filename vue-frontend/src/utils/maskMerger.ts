/**
 * 掩膜工具
 * 用于管理用户笔刷掩膜
 * 
 * 注意：掩膜合并逻辑已移至后端 inpainting.py 处理
 * 前端只负责：
 * 1. 创建初始用户掩膜
 * 2. 在用户使用笔刷时更新用户掩膜
 */

/**
 * 初始化用户掩膜（灰色画布）
 * @param width - 图片宽度
 * @param height - 图片高度
 * @returns Base64 编码的灰色掩膜图片
 */
export function createInitialUserMask(width: number, height: number): string {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')!

    // 填充灰色 (127)
    ctx.fillStyle = 'rgb(127, 127, 127)'
    ctx.fillRect(0, 0, width, height)

    // 转为 Base64（去掉 data URL 前缀）
    const dataUrl = canvas.toDataURL('image/png')
    return dataUrl.split(',')[1] || ''
}

/**
 * 更新用户掩膜 - 添加消除区域（白色）
 * @param currentUserMask - 当前用户掩膜 (Base64)
 * @param width - 图片宽度
 * @param height - 图片高度
 * @param path - 笔刷路径
 * @param radius - 笔刷半径
 * @returns 更新后的用户掩膜 (Base64)
 */
export async function addErasureToUserMask(
    currentUserMask: string | null | undefined,
    width: number,
    height: number,
    path: Array<{ x: number, y: number }>,
    radius: number
): Promise<string> {
    // 如果没有现有掩膜，先创建一个
    if (!currentUserMask) {
        currentUserMask = createInitialUserMask(width, height)
    }

    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')!

        const img = new Image()
        img.onload = () => {
            // 绘制现有掩膜（缩放到目标尺寸）
            ctx.drawImage(img, 0, 0, width, height)

            // 用白色绘制笔刷路径
            ctx.fillStyle = 'white'
            for (const pos of path) {
                ctx.beginPath()
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2)
                ctx.fill()
            }

            // 转为 Base64
            const dataUrl = canvas.toDataURL('image/png')
            const newMask = dataUrl.split(',')[1] || ''
            resolve(newMask)
        }
        img.onerror = reject
        img.src = 'data:image/png;base64,' + currentUserMask
    })
}

/**
 * 更新用户掩膜 - 添加还原区域（黑色）
 * @param currentUserMask - 当前用户掩膜 (Base64)
 * @param width - 图片宽度
 * @param height - 图片高度
 * @param path - 笔刷路径
 * @param radius - 笔刷半径
 * @returns 更新后的用户掩膜 (Base64)
 */
export async function addRestorationToUserMask(
    currentUserMask: string | null | undefined,
    width: number,
    height: number,
    path: Array<{ x: number, y: number }>,
    radius: number
): Promise<string> {
    // 如果没有现有掩膜，先创建一个
    if (!currentUserMask) {
        currentUserMask = createInitialUserMask(width, height)
    }

    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')!

        const img = new Image()
        img.onload = () => {
            // 绘制现有掩膜（缩放到目标尺寸）
            ctx.drawImage(img, 0, 0, width, height)

            // 用黑色绘制笔刷路径
            ctx.fillStyle = 'black'
            for (const pos of path) {
                ctx.beginPath()
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2)
                ctx.fill()
            }

            // 转为 Base64
            const dataUrl = canvas.toDataURL('image/png')
            const newMask = dataUrl.split(',')[1] || ''
            resolve(newMask)
        }
        img.onerror = reject
        img.src = 'data:image/png;base64,' + currentUserMask
    })
}
