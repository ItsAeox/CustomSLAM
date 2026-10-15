// Loaded first by every page. On static hosting (no COOP/COEP headers) it installs coi-sw.js and reloads once, so
// the page becomes cross-origin isolated (WASM threads). Does nothing when the server already sends the headers
// (web/serve.py) or the browser has no service workers.
(function () {
  if (window.crossOriginIsolated || !('serviceWorker' in navigator) || !window.isSecureContext) return;
  const swUrl = new URL('coi-sw.js', document.currentScript.src);
  navigator.serviceWorker.register(swUrl).then(function (reg) {
    const reloaded = sessionStorage.getItem('coiReloaded');
    if (navigator.serviceWorker.controller && reloaded) return;  // already controlled; nothing more to do
    if (reloaded) return;  // reloaded once already and still not isolated: give up (avoid a reload loop)
    sessionStorage.setItem('coiReloaded', '1');
    const go = function () { location.reload(); };
    if (reg.active) go(); else navigator.serviceWorker.addEventListener('controllerchange', go);
  }).catch(function (e) { console.error('coi service worker registration failed', e); });
})();
