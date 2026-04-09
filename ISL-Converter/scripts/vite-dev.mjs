import { createRequire } from 'node:module'

// Workaround for Windows environments where `net use` spawn fails (EPERM)
if (process.platform === 'win32') {
  const require = createRequire(import.meta.url)
  const childProcess = require('child_process')
  const originalExec = childProcess.exec

  childProcess.exec = (command, options, callback) => {
    const cmd = String(command || '').trim().toLowerCase()
    if (cmd === 'net use') {
      const cb = typeof options === 'function' ? options : callback
      if (typeof cb === 'function') {
        queueMicrotask(() => cb(null, ''))
      }
      return undefined
    }
    return originalExec(command, options, callback)
  }
}

try {
  const { createServer } = await import('vite')
  const react = (await import('@vitejs/plugin-react')).default
  const server = await createServer({
    root: process.cwd(),
    configFile: false,
    plugins: [react()],
    server: { host: true }
  })
  await server.listen()
  server.printUrls()

  const addr = server.httpServer?.address()
  if (addr && typeof addr === 'object') {
    console.log(`Vite running at http://localhost:${addr.port}/`)
  }

  // Keep process alive even if the event loop would otherwise exit
  setInterval(() => {}, 1 << 30)
} catch (err) {
  console.error('Vite dev server failed to start.')
  console.error(err)
  process.exit(1)
}
