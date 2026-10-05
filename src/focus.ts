import type {ImageDataLike} from './image.ts'
import {defaults, locate, type FocusPoint, type Options, type SaliencyMap} from './locate.ts'
import {infer} from './net.ts'
import {resize} from './resize.ts'
import {thumbnail} from './thumbnail.ts'
import {input} from './weights.ts'

/**
 * Compute the importance map with the learned model. The map has the
 * image's aspect, `size` pixels on its longest side.
 */
export function saliency(image: ImageDataLike, options: Options = {}): SaliencyMap {
  const {size} = {...defaults, ...options}
  const thumb = thumbnail(image, input)
  const scale = size / Math.max(image.width, image.height)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))
  // A blank image would only show the model's learned prior (subjects sit
  // high in the frame), so it gets an empty map and the centre
  if (uniform(thumb.rgb) && (!thumb.alpha || uniform(thumb.alpha)))
    return {width, height, data: new Float32Array(width * height)}
  const logits = infer(thumb.rgb, input)
  const side = input / 2
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  const map = logits.map(v => Math.exp(v - max))
  if (thumb.alpha) {
    const alpha = resize(thumb.alpha, input, input, side, side)
    for (let i = 0; i < map.length; i++) map[i]! *= alpha[i]!
  }
  const data = resize(map, side, side, width, height)
  let top = 0
  for (const v of data) if (v > top) top = v
  if (top) for (let i = 0; i < data.length; i++) data[i]! /= top
  return {width, height, data}
}

function uniform(data: Float32Array): boolean {
  for (const v of data) if (Math.abs(v - data[0]!) > 1e-4) return false
  return true
}

/** Find the main subject in an image and return its focus point. */
export function focusPoint(image: ImageDataLike, options: Options = {}): FocusPoint {
  const p = {...defaults, ...options}
  return locate(saliency(image, p), p)
}
