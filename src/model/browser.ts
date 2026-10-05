import type {FocusPoint, Options} from '../focus.ts'
import {pixels, type Source} from '../pixels.ts'
import {focusPoint} from './index.ts'

export * from './index.ts'

/**
 * Detect the focus point of a loaded image, canvas or bitmap with the
 * learned model. The model looks at a 64 px thumbnail, so drawing at 256 px
 * leaves the library enough pixels to average.
 */
export function focusPointFromImage(source: Source, options: Options = {}): FocusPoint {
  return focusPoint(pixels(source, 256), options)
}
