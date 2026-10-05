// Report the minified and gzipped size of each entry point.
import {gzipSync} from 'node:zlib'
import {rm, writeFile} from 'node:fs/promises'

for (const entry of ['index']) {
  // Reference every export so nothing is tree-shaken away
  const probe = `scripts/.size-${entry.replace('/', '-')}.ts`
  await writeFile(probe, `import * as m from '../src/${entry}.ts'\n;(globalThis as any).m = m\n`)
  const result = await Bun.build({entrypoints: [probe], minify: true, target: 'browser'})
  await rm(probe)
  const code = await result.outputs[0]!.text()
  const gz = gzipSync(code, {level: 9}).length
  console.log(`${entry.padEnd(12)} ${(code.length / 1024).toFixed(2)} kB min, ${(gz / 1024).toFixed(2)} kB gzip`)
}
