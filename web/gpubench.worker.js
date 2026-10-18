// gpubench.html: the benchmark in a dedicated worker, where the SLAM runs (slam.worker.js)
import { runBench } from './gpubench.js';

onmessage = async (e) => {
  try {
    const r = await runBench((line) => postMessage({ line }), e.data.iters || 150);
    postMessage({ done: r });
  } catch (err) {
    postMessage({ error: String(err.message || err) });
  }
};
