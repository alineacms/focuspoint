/**
 * Bilinear resize of a single-channel map, sampling at pixel centres (as
 * PyTorch's `interpolate(align_corners=False)` does).
 */
export function resize(
  data: Float32Array,
  width: number,
  height: number,
  outWidth: number,
  outHeight: number
): Float32Array {
  const out = new Float32Array(outWidth * outHeight)
  const sx = width / outWidth
  const sy = height / outHeight
  for (let y = 0; y < outHeight; y++) {
    const fy = Math.min(height - 1, Math.max(0, (y + 0.5) * sy - 0.5))
    const y0 = Math.floor(fy)
    const y1 = Math.min(height - 1, y0 + 1)
    const ty = fy - y0
    for (let x = 0; x < outWidth; x++) {
      const fx = Math.min(width - 1, Math.max(0, (x + 0.5) * sx - 0.5))
      const x0 = Math.floor(fx)
      const x1 = Math.min(width - 1, x0 + 1)
      const tx = fx - x0
      const top = data[y0 * width + x0]! * (1 - tx) + data[y0 * width + x1]! * tx
      const bottom = data[y1 * width + x0]! * (1 - tx) + data[y1 * width + x1]! * tx
      out[y * outWidth + x] = top * (1 - ty) + bottom * ty
    }
  }
  return out
}
