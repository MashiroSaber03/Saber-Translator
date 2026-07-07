type BrushPoint = {
  x: number
  y: number
}

const INITIAL_MASK_COLOR = 'rgb(127, 127, 127)'
const MASK_DATA_URL_PREFIX = 'data:image/png;base64,'

function maskCanvasToBase64(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png').split(',')[1] || ''
}

function createMaskCanvas(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return [canvas, canvas.getContext('2d')!]
}

export function createInitialUserMask(width: number, height: number): string {
  const [canvas, context] = createMaskCanvas(width, height)
  context.fillStyle = INITIAL_MASK_COLOR
  context.fillRect(0, 0, width, height)
  return maskCanvasToBase64(canvas)
}

function paintUserMask(
  currentUserMask: string | null | undefined,
  width: number,
  height: number,
  path: BrushPoint[],
  radius: number,
  brushColor: 'white' | 'black',
): Promise<string> {
  const sourceMask = currentUserMask || createInitialUserMask(width, height)

  return new Promise((resolve, reject) => {
    const [canvas, context] = createMaskCanvas(width, height)
    const image = new Image()

    image.onload = () => {
      context.drawImage(image, 0, 0, width, height)
      context.fillStyle = brushColor

      for (const point of path) {
        context.beginPath()
        context.arc(point.x, point.y, radius, 0, Math.PI * 2)
        context.fill()
      }

      resolve(maskCanvasToBase64(canvas))
    }

    image.onerror = reject
    image.src = `${MASK_DATA_URL_PREFIX}${sourceMask}`
  })
}

export async function addErasureToUserMask(
  currentUserMask: string | null | undefined,
  width: number,
  height: number,
  path: BrushPoint[],
  radius: number,
): Promise<string> {
  return paintUserMask(currentUserMask, width, height, path, radius, 'white')
}

export async function addRestorationToUserMask(
  currentUserMask: string | null | undefined,
  width: number,
  height: number,
  path: BrushPoint[],
  radius: number,
): Promise<string> {
  return paintUserMask(currentUserMask, width, height, path, radius, 'black')
}
