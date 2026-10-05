import {defaults, focusPoint, type FocusPoint, type Options} from './focus.ts'
import {pixels, type Source} from './pixels.ts'

export * from './index.ts'

/**
 * Detect the focus point of a loaded image, canvas or bitmap. The source is
 * drawn onto a small canvas first so the browser does the heavy downscaling.
 */
export function focusPointFromImage(source: Source, options: Options = {}): FocusPoint {
  // Draw at twice the working size, the library area-averages the rest
  return focusPoint(pixels(source, (options.size ?? defaults.size) * 2), options)
}
