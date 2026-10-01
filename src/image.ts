/** Anything shaped like ImageData: RGBA bytes in row-major order. */
export interface ImageDataLike {
  data: ArrayLike<number>
  width: number
  height: number
}

/** A small image in CIE Lab, one plane per channel. */
export interface LabImage {
  width: number
  height: number
  l: Float32Array
  a: Float32Array
  b: Float32Array
  /** Opacity 0..1, or undefined when the image is fully opaque. */
  alpha: Float32Array | undefined
}

let linear: Float32Array | undefined
function srgbToLinear(): Float32Array {
  if (linear) return linear
  linear = new Float32Array(256)
  for (let i = 0; i < 256; i++) {
    const c = i / 255
    linear[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  }
  return linear
}

function labF(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116
}

/**
 * Area-average the image down so its longest side is at most `size` pixels
 * and convert it to Lab. Averaging happens in linear light. Very large inputs
 * are sampled on a sparse grid (~16 samples per output pixel) to stay fast.
 */
export function toLab(image: ImageDataLike, size: number): LabImage {
  const {data, width: sw, height: sh} = image
  if (sw < 1 || sh < 1 || data.length < sw * sh * 4)
    throw new Error('focuspoint: invalid image data')
  const scale = Math.min(1, size / Math.max(sw, sh))
  const width = Math.max(1, Math.round(sw * scale))
  const height = Math.max(1, Math.round(sh * scale))
  const n = width * height
  const step = Math.max(1, Math.floor(1 / (scale * 4)))
  const lut = srgbToLinear()
  const r = new Float32Array(n)
  const g = new Float32Array(n)
  const b = new Float32Array(n)
  const a = new Float32Array(n)
  const count = new Float32Array(n)
  const fx = width / sw
  const fy = height / sh
  const column = new Int32Array(sw)
  for (let x = 0; x < sw; x += step) column[x] = Math.min(width - 1, Math.floor(x * fx))
  let translucent = false
  for (let y = 0; y < sh; y += step) {
    const row = Math.min(height - 1, Math.floor(y * fy)) * width
    for (let x = 0; x < sw; x += step) {
      const i = (y * sw + x) * 4
      const o = row + column[x]!
      const alpha = data[i + 3]! / 255
      if (alpha < 1) translucent = true
      // Premultiply so transparent pixels don't leak colour
      r[o]! += lut[data[i]!]! * alpha
      g[o]! += lut[data[i + 1]!]! * alpha
      b[o]! += lut[data[i + 2]!]! * alpha
      a[o]! += alpha
      count[o]!++
    }
  }
  const L = new Float32Array(n)
  const A = new Float32Array(n)
  const B = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const c = count[i]!
    const opacity = c ? a[i]! / c : 0
    a[i] = opacity
    // Composite over mid grey: transparent areas read as flat background
    const bg = 0.2158 * (1 - opacity)
    const R = c ? r[i]! / c + bg : bg
    const G = c ? g[i]! / c + bg : bg
    const Bl = c ? b[i]! / c + bg : bg
    const X = (0.4124 * R + 0.3576 * G + 0.1805 * Bl) / 0.95047
    const Y = 0.2126 * R + 0.7152 * G + 0.0722 * Bl
    const Z = (0.0193 * R + 0.1192 * G + 0.9505 * Bl) / 1.08883
    const fX = labF(X)
    const fY = labF(Y)
    const fZ = labF(Z)
    L[i] = 116 * fY - 16
    A[i] = 500 * (fX - fY)
    B[i] = 200 * (fY - fZ)
  }
  return {width, height, l: L, a: A, b: B, alpha: translucent ? a : undefined}
}

/** Halve an image's resolution by averaging 2x2 blocks. */
export function half(img: LabImage): LabImage {
  const width = Math.max(1, img.width >> 1)
  const height = Math.max(1, img.height >> 1)
  const shrink = (src: Float32Array) => {
    const out = new Float32Array(width * height)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const x0 = Math.min(img.width - 1, x * 2)
        const y0 = Math.min(img.height - 1, y * 2)
        const x1 = Math.min(img.width - 1, x0 + 1)
        const y1 = Math.min(img.height - 1, y0 + 1)
        out[y * width + x] =
          (src[y0 * img.width + x0]! +
            src[y0 * img.width + x1]! +
            src[y1 * img.width + x0]! +
            src[y1 * img.width + x1]!) /
          4
      }
    return out
  }
  return {
    width,
    height,
    l: shrink(img.l),
    a: shrink(img.a),
    b: shrink(img.b),
    alpha: img.alpha && shrink(img.alpha)
  }
}
