// Cross-origin isolation for static hosting (GitHub Pages cannot send headers): this service worker re-serves every
// response with COOP/COEP (+ CORP), which browsers require for SharedArrayBuffer = WebAssembly threads.
// Same technique as the MIT-licensed "coi-serviceworker" (gzuidhof); written from scratch for this project.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;  // opaque response: pass through
      const headers = new Headers(res.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }).catch((err) => {
      console.error('coi-sw fetch failed', err);
      return Response.error();
    }),
  );
});
