// WebXR version of the blind study (claude/STUDY.md §9): Chrome's immersive-ar session (ARCore underneath), placed with
// WebXR hit testing at the tapped screen point and kept with an anchor when the browser supports anchors (as an ARCore
// app would). Same plant, lights, size rule (0.6 x the camera's height above the hit surface) and task overlay as the
// other versions; the overlay is shown with the DOM-overlay feature. Tracking = a viewer pose that is not emulated.
import * as THREE from 'three';
import { Overlay, finishCondition, startGate, versionIndex } from './common.js';
import { TREE_HEIGHT, loadTree, makeScene } from './scene.js';

startGate(versionIndex(), async () => {
  let ov = null;
  try {
    const session = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['hit-test', 'local'], optionalFeatures: ['dom-overlay', 'anchors'], domOverlay: { root: document.body } });
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(innerWidth, innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType('local');
    document.body.append(renderer.domElement);
    await renderer.xr.setSession(session);
    const ref = renderer.xr.getReferenceSpace();
    const { scene, sun } = makeScene();
    const cam = new THREE.PerspectiveCamera();
    const proto = await loadTree(new URL('../assets/tree.glb', import.meta.url).href);
    let tree = null, anchor = null, tracking = false, pendingTap = null;

    const xrCam = () => renderer.xr.getCamera().cameras[0] || renderer.xr.getCamera();
    ov = new Overlay({
      objectScreen: () => {
        if (!tree) return null;
        const v = tree.position.clone().project(xrCam());
        return { x: (v.x + 1) / 2 * innerWidth, y: (1 - v.y) / 2 * innerHeight, visible: tree.visible && v.z > -1 && v.z < 1 };
      },
    }, async (log) => { try { await session.end(); } catch { /* already ended */ } finishCondition('webxr', { ...log, anchors: !!anchor }); });

    // taps on the camera view (DOM overlay): the overlay first (return check), else a one-shot hit test along the tap ray
    document.body.addEventListener('pointerdown', (e) => {
      if (e.target.closest && e.target.closest('#studyOverlay')) return;
      if (ov.tap(e.clientX, e.clientY) || !tracking) return;
      pendingTap = { x: e.clientX, y: e.clientY };
    });
    session.addEventListener('end', () => { tracking = false; });

    renderer.setAnimationLoop((t, frame) => {  // synchronous: rendering must stay inside the XR frame
      if (!frame) return;
      const vp = frame.getViewerPose(ref);
      tracking = !!vp && !vp.emulatedPosition;
      if (pendingTap && tracking) {
        const tap = pendingTap; pendingTap = null;
        const c = xrCam();
        const ndc = new THREE.Vector3((tap.x / innerWidth) * 2 - 1, -(tap.y / innerHeight) * 2 + 1, 0.5);
        const dir = ndc.unproject(c).sub(c.position).normalize();
        const o = c.position.clone();
        const once = (src) => (t2, f2) => {
          const hits = f2.getHitTestResults(src);
          src.cancel();
          if (!hits.length) return;
          const hit = hits[0], pose = hit.getPose(ref);
          if (!pose) return;
          const m = new THREE.Matrix4().fromArray(pose.transform.matrix);
          const p = new THREE.Vector3().setFromMatrixPosition(m), n = new THREE.Vector3(0, 1, 0).transformDirection(m);
          if (tree) scene.remove(tree);
          if (anchor) { anchor.delete?.(); anchor = null; }
          tree = proto.clone();
          tree.position.copy(p);
          tree.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);
          tree.scale.setScalar(TREE_HEIGHT(Math.max(0.05, o.clone().sub(p).dot(n))));
          scene.add(tree);
          if (hit.createAnchor) hit.createAnchor().then((a) => { anchor = a; }).catch(() => {});
          ov.placed();
        };
        session.requestHitTestSource({
          space: ref, offsetRay: new XRRay({ x: o.x, y: o.y, z: o.z, w: 1 }, { x: dir.x, y: dir.y, z: dir.z, w: 0 }) })
          .then((src) => session.requestAnimationFrame(once(src))).catch(() => {});
      }
      if (tree && anchor) {
        const ap = frame.getPose(anchor.anchorSpace, ref);
        if (ap) { const m = new THREE.Matrix4().fromArray(ap.transform.matrix); tree.position.setFromMatrixPosition(m); tree.quaternion.setFromRotationMatrix(m); }
      }
      if (tree) tree.visible = tracking;  // hidden while not tracking, as the other versions
      if (vp) { const p = vp.transform.position; sun.position.set(p.x, p.y + 2, p.z); }
      renderer.render(scene, cam);
      ov.frame(tracking);
    });
  } catch (e) {
    ov ??= new Overlay({}, (log) => finishCondition('webxr', log));
    ov.fail(String(e.message || e));
  }
});
