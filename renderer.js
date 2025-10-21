let canvas, ctx2d, hudEl;

export async function initRenderer(cvs) {
  canvas = cvs;
  ctx2d = canvas.getContext('2d', { alpha: true });
  // Make sure the canvas itself is transparent; video sits behind it.
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  // HUD
  hudEl = document.getElementById('hud');
  if (!hudEl) {
    hudEl = document.createElement('div');
    hudEl.id = 'hud';
    hudEl.style.cssText = 'position:fixed;right:8px;top:8px;color:#0f0;font:12px monospace;z-index:20;text-align:right';
    document.body.appendChild(hudEl);
  }
  updateHUDText('…');
}

/**
 * pts: JS array [x0,y0, x1,y1, ...] in pixel coords (origin at top-left)
 */
export function drawPoints(pts, W, H) {
  if (!ctx2d) return;

  // Clear the canvas
  ctx2d.clearRect(0, 0, W, H);

  // Draw points as tiny white squares (faster than arcs; crisp even when scaled)
  ctx2d.fillStyle = '#ffffff';
  const n = (pts.length / 2) | 0;
  const s = 2; // point size in pixels
  for (let i = 0; i < n; ++i) {
    const x = pts[2*i], y = pts[2*i + 1];
    // Clamp to bounds just in case
    if (x >= 0 && y >= 0 && x < W && y < H) {
      ctx2d.fillRect(x - (s>>1), y - (s>>1), s, s);
    }
  }
}

export function updateHUDText(t) {
  if (hudEl) hudEl.textContent = t || '';
}
