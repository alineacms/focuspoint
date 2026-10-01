// Bundle demo/main.ts and inline it into a single self-contained HTML file.
// Usage: bun scripts/demo.ts  ->  demo/focuspoint-viewer.html
const result = await Bun.build({entrypoints: ['demo/main.ts'], minify: true, target: 'browser', format: 'iife'})
if (!result.success) {
  for (const log of result.logs) console.error(log)
  process.exit(1)
}
const js = (await result.outputs[0]!.text()).replaceAll('</script', '<\\/script')
const html = (await Bun.file('demo/template.html').text()).replace('/*SCRIPT*/', () => js)
await Bun.write('demo/focuspoint-viewer.html', html)
console.log(`demo/focuspoint-viewer.html ${(html.length / 1024).toFixed(1)} kB`)
