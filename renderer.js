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
  ctx2d.fillStyle = '#ffe658ff';
  const n = (pts.length / 2) | 0;
  const s = 3; // point size in pixels
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

// === Attitude overlay (yaw/pitch/roll) ======================================
export function drawAttitude(yawDeg, pitchDeg, rollDeg, W, H) {
  if (!ctx2d || !Number.isFinite(yawDeg) || !Number.isFinite(pitchDeg) || !Number.isFinite(rollDeg)) return;

  // Tunable scales
  const degPerHalfScreen = 30;               // ±30° pitch spans ~half the screen
  const pitchPxPerDeg = (H * 0.5) / degPerHalfScreen;

  // Convert to radians (roll rotates the horizon)
  const rollRad = rollDeg * Math.PI / 180;
  const pitchPx = pitchDeg * pitchPxPerDeg;

  ctx2d.save();

  // semi-transparent layer so points remain visible underneath
  ctx2d.globalAlpha = 0.9;

  // Move to center, rotate by -roll (standard artificial horizon convention)
  ctx2d.translate(W * 0.5, H * 0.5);
  ctx2d.rotate(-rollRad);

  // Draw horizon line (shifted by pitch)
  ctx2d.beginPath();
  ctx2d.moveTo(-W,  pitchPx);
  ctx2d.lineTo( +W, pitchPx);
  ctx2d.strokeStyle = '#00ffaa';
  ctx2d.lineWidth = 2;
  ctx2d.stroke();

  // Pitch ladder ticks every 10°
  ctx2d.strokeStyle = '#00ffaa';
  ctx2d.fillStyle = '#00ffaa';
  ctx2d.lineWidth = 1;

  for (let d = -60; d <= 60; d += 10) {
    const y = (d * pitchPxPerDeg) + pitchPx;
    const len = (d % 20 === 0) ? 40 : 24;
    ctx2d.beginPath();
    ctx2d.moveTo(-len, y);
    ctx2d.lineTo( len, y);
    ctx2d.stroke();

    // Label major lines (every 20°)
    if (d % 20 === 0 && d !== 0) {
      const txt = `${d > 0 ? '+' : ''}${d}°`;
      ctx2d.font = '12px monospace';
      ctx2d.textAlign = 'center';
      ctx2d.textBaseline = 'middle';
      ctx2d.fillText(txt, 0, y - 10);
    }
  }

  // Center marker (simple reticle)
  ctx2d.beginPath();
  ctx2d.moveTo(-10, 0);
  ctx2d.lineTo( 10, 0);
  ctx2d.moveTo(0, -10);
  ctx2d.lineTo(0,  10);
  ctx2d.strokeStyle = '#ffffff';
  ctx2d.lineWidth = 1.5;
  ctx2d.stroke();

  ctx2d.restore();

  // Heading tape (yaw) across the top of the screen (not rotated)
  const tapeH = 22;
  const tapeY = 8;
  const tickStepDeg = 10;  // small ticks
  const majorEvery = 30;   // major ticks/labels
  const pxPerDeg = W / 180; // ~180° span across width

  // Wrap yaw to [0, 360)
  const yaw360 = ((yawDeg % 360) + 360) % 360;

  ctx2d.save();
  ctx2d.globalAlpha = 0.85;
  ctx2d.fillStyle = '#000000';
  ctx2d.fillRect(0, tapeY, W, tapeH);
  ctx2d.strokeStyle = '#0f0';
  ctx2d.strokeRect(0.5, tapeY + 0.5, W - 1, tapeH - 1);

  // Draw ticks centered at current yaw
  const centerX = W * 0.5;
  const startDeg = Math.floor(yaw360 - (W * 0.5) / pxPerDeg);
  const endDeg   = Math.ceil (yaw360 + (W * 0.5) / pxPerDeg);

  ctx2d.strokeStyle = '#66ccff';
  ctx2d.fillStyle   = '#66ccff';
  ctx2d.lineWidth = 1;

  for (let d = startDeg; d <= endDeg; d += tickStepDeg) {
    // Wrap each tick label
    const dd = ((d % 360) + 360) % 360;
    const x = centerX + (d - yaw360) * pxPerDeg;

    const isMajor = (dd % majorEvery) === 0;
    const h = isMajor ? tapeH - 6 : tapeH - 10;

    ctx2d.beginPath();
    ctx2d.moveTo(x + 0.5, tapeY + tapeH);
    ctx2d.lineTo(x + 0.5, tapeY + h);
    ctx2d.stroke();

    if (isMajor) {
      ctx2d.font = '11px monospace';
      ctx2d.textAlign = 'center';
      ctx2d.textBaseline = 'bottom';
      ctx2d.fillText(String(dd), x, tapeY + h - 1);
    }
  }

  // Current heading readout
  ctx2d.font = '12px monospace';
  ctx2d.textAlign = 'center';
  ctx2d.textBaseline = 'middle';
  ctx2d.fillStyle = '#ffffff';
  ctx2d.fillText(`${yaw360.toFixed(0)}°`, centerX, tapeY + tapeH * 0.5);

  // Center caret
  ctx2d.beginPath();
  ctx2d.moveTo(centerX, tapeY + tapeH);
  ctx2d.lineTo(centerX - 6, tapeY + tapeH - 6);
  ctx2d.lineTo(centerX + 6, tapeY + tapeH - 6);
  ctx2d.closePath();
  ctx2d.fillStyle = '#ffffff';
  ctx2d.fill();

  ctx2d.restore();
}


export function drawPathXZ(flatXZ, W, H) {
  if (!flatXZ || !flatXZ.length) return;
  // Inset box
  const boxW = Math.round(W * 0.35);
  const boxH = Math.round(H * 0.35);
  const margin = 8;
  const x0 = margin, y0 = H - boxH - margin;

  // Compute bounds
  let minX=Infinity, maxX=-Infinity, minZ=Infinity, maxZ=-Infinity;
  for (let i = 0; i < flatXZ.length; i += 2) {
    const x = flatXZ[i], z = flatXZ[i+1];
    if (!Number.isFinite(x) || !Number.isFinite(z)) continue;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (!isFinite(minX) || !isFinite(minZ)) return;

  // Pad bounds a bit so path is not glued to edges
  const pad = 1e-3;
  minX -= pad; maxX += pad; minZ -= pad; maxZ += pad;

  // Map (x,z) -> inset pixel
  const sx = (maxX - minX) > 1e-6 ? (boxW - 12) / (maxX - minX) : 1.0;
  const sz = (maxZ - minZ) > 1e-6 ? (boxH - 12) / (maxZ - minZ) : 1.0;

  // Background
  ctx2d.save();
  ctx2d.globalAlpha = 0.6;
  ctx2d.fillStyle = '#000';
  ctx2d.fillRect(x0, y0, boxW, boxH);
  ctx2d.globalAlpha = 1.0;
  ctx2d.strokeStyle = '#0f0';
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(x0+0.5, y0+0.5, boxW-1, boxH-1);

  // Draw path
  ctx2d.beginPath();
  for (let i = 0; i < flatXZ.length; i += 2) {
    const X = flatXZ[i], Z = flatXZ[i+1];
    const px = x0 + 6 + (X - minX) * sx;
    const pz = y0 + boxH - 6 - (Z - minZ) * sz; // z up in inset
    if (i === 0) ctx2d.moveTo(px, pz); else ctx2d.lineTo(px, pz);
  }
  ctx2d.strokeStyle = '#66ccff'; // path color
  ctx2d.lineWidth = 2;
  ctx2d.stroke();

  // Draw current camera as a small triangle pointing forward (approx)
  if (flatXZ.length >= 4) {
    const X = flatXZ[flatXZ.length-2], Z = flatXZ[flatXZ.length-1];
    const px = x0 + 6 + (X - minX) * sx;
    const pz = y0 + 6 + (Z - minZ) * sz; // forward (+Z) draws downward
    ctx2d.beginPath();
    ctx2d.arc(px, pz, 3, 0, Math.PI*2);
    ctx2d.fillStyle = '#ffffff';
    ctx2d.fill();
  }
  ctx2d.restore();
}
