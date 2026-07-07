import type { PipelineRuntime, TaskContext } from './runtime'

export function resolveTaskStyleFields(
  context: TaskContext,
  runtime: PipelineRuntime
): Record<string, unknown> {
  const image = context.sourceImage
  const saved = runtime.savedTextStyles
  const { textStyle } = runtime.settingsSnapshot

  return {
    fontSize: image.fontSize ?? saved?.fontSize ?? textStyle.fontSize,
    autoFontSize: image.autoFontSize ?? saved?.autoFontSize ?? textStyle.autoFontSize,
    fontFamily: image.fontFamily ?? saved?.fontFamily ?? textStyle.fontFamily,
    layoutDirection: image.layoutDirection ?? saved?.layoutDirection ?? textStyle.layoutDirection,
    textColor: image.textColor ?? saved?.textColor ?? textStyle.textColor,
    fillColor: image.fillColor ?? saved?.fillColor ?? textStyle.fillColor,
    strokeEnabled: image.strokeEnabled ?? saved?.strokeEnabled ?? textStyle.strokeEnabled,
    strokeColor: image.strokeColor ?? saved?.strokeColor ?? textStyle.strokeColor,
    strokeWidth: image.strokeWidth ?? saved?.strokeWidth ?? textStyle.strokeWidth,
    lineSpacing: image.lineSpacing ?? saved?.lineSpacing ?? textStyle.lineSpacing,
    textAlign: image.textAlign ?? saved?.textAlign ?? textStyle.textAlign,
    inpaintMethod: image.inpaintMethod ?? saved?.inpaintMethod ?? textStyle.inpaintMethod,
    useAutoTextColor: image.useAutoTextColor ?? saved?.useAutoTextColor ?? textStyle.useAutoTextColor,
  }
}
