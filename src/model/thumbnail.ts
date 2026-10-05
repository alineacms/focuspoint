import {srgbToLinear, type ImageDataLike} from '../image.ts'

export interface Thumbnail {
  size: number
  /** sRGB 0..1, one plane per channel */
  rgb: Float32Array
  /** Opacity 0..1, or undefined when the image is fully opaque */
  alpha: Float32Array | undefined
}

/**
 * Squash the image to size x size, whatever its aspect, area-averaging in
 * linear light. Large inputs are sampled on a sparse grid (~16 samples per
 * output pixel); cells no source pixel falls in (small inputs) take the
 * nearest pixel.
 */
export function thumbnail(image: ImageDataLike, size: number): Thumbnail {
  const {data, width: sw, height: sh} = image
  if (sw < 1 || sh < 1 || data.length < sw * sh * 4) throw new Error('focuspoint: invalid image data')
  const n = size * size
  const fx = size / sw
  const fy = size / sh
  const step = Math.max(1, Math.floor(Math.min(1 / fx, 1 / fy) / 4))
  const lut = srgbToLinear()
  const sums = new Float32Array(n * 4)
  const count = new Float32Array(n)
  let translucent = false
  const add = (i: number, o: number) => {
    const alpha = data[i + 3]! / 255
    if (alpha < 1) translucent = true
    // Premultiply so transparent pixels don't leak colour
    sums[o * 4]! += lut[data[i]!]! * alpha
    sums[o * 4 + 1]! += lut[data[i + 1]!]! * alpha
    sums[o * 4 + 2]! += lut[data[i + 2]!]! * alpha
    sums[o * 4 + 3]! += alpha
    count[o]!++
  }
  for (let y = 0; y < sh; y += step) {
    const row = Math.min(size - 1, Math.floor(y * fy)) * size
    for (let x = 0; x < sw; x += step) add((y * sw + x) * 4, row + Math.min(size - 1, Math.floor(x * fx)))
  }
  for (let o = 0; o < n; o++)
    if (!count[o]) {
      const x = Math.min(sw - 1, Math.floor(((o % size) + 0.5) / fx))
      const y = Math.min(sh - 1, Math.floor((Math.floor(o / size) + 0.5) / fy))
      add((y * sw + x) * 4, o)
    }
  const rgb = new Float32Array(n * 3)
  const alpha = new Float32Array(n)
  for (let o = 0; o < n; o++) {
    const opacity = sums[o * 4 + 3]! / count[o]!
    alpha[o] = opacity
    // Composite over mid grey: transparent areas read as flat background
    const bg = 0.2158 * (1 - opacity)
    for (let c = 0; c < 3; c++) rgb[c * n + o] = toSrgb(sums[o * 4 + c]! / count[o]! + bg)
  }
  return {size, rgb, alpha: translucent ? alpha : undefined}
}

function toSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055
}
