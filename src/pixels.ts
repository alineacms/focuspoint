export type Source = HTMLImageElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas | HTMLVideoElement

function dimensions(source: Source): [number, number] {
  if ('naturalWidth' in source) return [source.naturalWidth, source.naturalHeight]
  if ('videoWidth' in source) return [source.videoWidth, source.videoHeight]
  return [source.width, source.height]
}

/**
 * Draw a loaded image, canvas or bitmap onto a small canvas, at most `size`
 * pixels on its longest side, so the browser does the heavy downscaling.
 */
export function pixels(source: Source, size: number): ImageData {
  const [w, h] = dimensions(source)
  if (!w || !h) throw new Error('focuspoint: image has no dimensions, is it loaded?')
  const scale = Math.min(1, size / Math.max(w, h))
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
  return ctx.getImageData(0, 0, width, height)
}
