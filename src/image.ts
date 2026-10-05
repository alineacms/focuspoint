/** Anything shaped like ImageData: RGBA bytes in row-major order. */
export interface ImageDataLike {
  data: ArrayLike<number>
  width: number
  height: number
}

let linear: Float32Array | undefined
export function srgbToLinear(): Float32Array {
  if (linear) return linear
  linear = new Float32Array(256)
  for (let i = 0; i < 256; i++) {
    const c = i / 255
    linear[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  }
  return linear
}
