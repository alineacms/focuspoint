import {defaults as heuristic, locate, type FocusPoint, type Options, type Params, type SaliencyMap} from '../focus.ts'
import type {ImageDataLike} from '../image.ts'
import {infer} from './net.ts'
import {resize} from './resize.ts'
import {thumbnail} from './thumbnail.ts'
import {input} from './weights.ts'

export type {Box, FocusPoint, Options, Params, SaliencyMap} from '../focus.ts'
export type {ImageDataLike} from '../image.ts'
export {locate}

/** Subject selection and placement parameters, fitted for the model's map */
export const defaults: Params = {...heuristic, protect: 0.5}

/**
 * Compute the importance map with the learned model. The map has the
 * image's aspect, `size` pixels on its longest side.
 */
export function saliency(image: ImageDataLike, options: Options = {}): SaliencyMap {
  const {size} = {...defaults, ...options}
  const thumb = thumbnail(image, input)
  const logits = infer(thumb.rgb, input)
  const side = input / 2
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  const map = logits.map(v => Math.exp(v - max))
  if (thumb.alpha) {
    const alpha = resize(thumb.alpha, input, input, side, side)
    for (let i = 0; i < map.length; i++) map[i]! *= alpha[i]!
  }
  const scale = size / Math.max(image.width, image.height)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))
  const data = resize(map, side, side, width, height)
  let top = 0
  for (const v of data) if (v > top) top = v
  if (top) for (let i = 0; i < data.length; i++) data[i]! /= top
  return {width, height, data}
}

/** Find the main subject in an image with the learned model. */
export function focusPoint(image: ImageDataLike, options: Options = {}): FocusPoint {
  const p = {...defaults, ...options}
  return locate(saliency(image, p), p)
}
