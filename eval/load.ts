import sharp from 'sharp'

/** Decode an image file into RGBA pixels, downscaled to at most `max` px. */
export async function load(file: string, max = 256) {
  const {data, info} = await sharp(file)
    .rotate()
    .resize(max, max, {fit: 'inside', withoutEnlargement: true})
    .ensureAlpha()
    .raw()
    .toBuffer({resolveWithObject: true})
  return {data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), width: info.width, height: info.height}
}
