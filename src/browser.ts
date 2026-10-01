import {defaults, focusPoint, type FocusPoint, type Options} from './focus.ts'

export * from './index.ts'

type Source = HTMLImageElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas | HTMLVideoElement

function dimensions(source: Source): [number, number] {
  if ('naturalWidth' in source) return [source.naturalWidth, source.naturalHeight]
  if ('videoWidth' in source) return [source.videoWidth, source.videoHeight]
  return [source.width, source.height]
}

/**
 * Detect the focus point of a loaded image, canvas or bitmap. The source is
 * drawn onto a small canvas first so the browser does the heavy downscaling.
 */
export function focusPointFromImage(source: Source, options: Options = {}): FocusPoint {
  const [w, h] = dimensions(source)
  if (!w || !h) throw new Error('focuspoint: image has no dimensions, is it loaded?')
  // Draw at twice the working size, the library area-averages the rest
  const target = (options.size ?? defaults.size) * 2
  const scale = Math.min(1, target / Math.max(w, h))
  const width = Math.max(1, Math.round(w * scale))
  const height = Math.max(1, Math.round(h * scale))
  const canvas =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement('canvas'), {width, height})
  const ctx = canvas.getContext('2d', {willReadFrequently: true}) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null
  if (!ctx) throw new Error('focuspoint: 2d canvas is not available')
  ctx.drawImage(source, 0, 0, width, height)
  return focusPoint(ctx.getImageData(0, 0, width, height), options)
}
