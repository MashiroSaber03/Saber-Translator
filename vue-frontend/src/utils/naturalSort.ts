/**
 * 自然排序工具函数
 * 
 * 实现效果：
 * - file1.jpg, file2.jpg, file10.jpg → 按 1, 2, 10 排序
 * - 第1话/001.jpg, 第2话/001.jpg, 第10话/001.jpg → 按 1, 2, 10 排序
 * 
 * 参考: manga-translator-ui/file_service.py 的 _natural_sort_key
 */

/**
 * 生成自然排序的键
 * @param path 文件路径或文件名
 * @returns 用于排序的键数组
 */
export function naturalSortKey(path: string): Array<[boolean, number | string]> {
    // 规范化路径分隔符
    const normalizedPath = path.replace(/\\/g, '/')

    // 将路径分割成文本和数字部分
    const parts: Array<[boolean, number | string]> = []
    const regex = /(\d+)/g
    let lastIndex = 0
    let match: RegExpExecArray | null

    while ((match = regex.exec(normalizedPath)) !== null) {
        // 添加数字前的文本部分
        if (match.index > lastIndex) {
            const textPart = normalizedPath.slice(lastIndex, match.index)
            if (textPart) {
                parts.push([true, textPart.toLowerCase()])
            }
        }
        // 添加数字部分
        parts.push([false, parseInt(match[0], 10)])
        lastIndex = regex.lastIndex
    }

    // 添加最后的文本部分
    if (lastIndex < normalizedPath.length) {
        const textPart = normalizedPath.slice(lastIndex)
        if (textPart) {
            parts.push([true, textPart.toLowerCase()])
        }
    }

    return parts
}

/**
 * 自然排序比较函数
 * @param a 第一个路径
 * @param b 第二个路径
 * @returns 比较结果 (-1, 0, 1)
 */
export function naturalSortCompare(a: string, b: string): number {
    const keyA = naturalSortKey(a)
    const keyB = naturalSortKey(b)

    const minLength = Math.min(keyA.length, keyB.length)

    for (let i = 0; i < minLength; i++) {
        const [isTextA, valA] = keyA[i]!
        const [isTextB, valB] = keyB[i]!

        // 如果类型不同：数字排在文本前面
        if (isTextA !== isTextB) {
            return isTextA ? 1 : -1
        }

        // 同类型比较
        if (valA < valB) return -1
        if (valA > valB) return 1
    }

    // 长度不同时，短的排前面
    return keyA.length - keyB.length
}

/**
 * 对文件列表进行自然排序
 * @param files 文件数组
 * @param getPath 获取排序路径的函数（可选）
 * @returns 排序后的数组（原数组不变）
 */
export function naturalSort<T>(
    files: T[],
    getPath: (item: T) => string = (item) => String(item)
): T[] {
    return [...files].sort((a, b) => naturalSortCompare(getPath(a), getPath(b)))
}
