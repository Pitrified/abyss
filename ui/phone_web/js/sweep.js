// The rate sweep: one splat drawn at several counts in turn, with tracking
// running, each held long enough to settle and then measured, one log line per
// count and a table at the end. One page load gives the whole table, so a
// person on the phone copies the log once per delegate.
//
// Rates are placed in tiers rather than reported to the millisecond: what a
// count delivers is 30, 10 or 1 frames a second, and the cut-offs between those
// are TIER_30_FPS and TIER_10_FPS.

import { log } from "./state.js";

export const SETTLE_MS = 3000;
export const MEASURE_MS = 7000;
export const TIER_30_FPS = 20;
export const TIER_10_FPS = 5;

export const tier = (fps) => (fps >= TIER_30_FPS ? 30 : fps >= TIER_10_FPS ? 10 : 1);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// `useKeep(keep)` redraws the splat at that share and resolves to its count;
// `restartTiming()` starts fresh medians; `timing()` reads them.
export async function runSweep(keeps, { useKeep, restartTiming, timing }) {
  const results = [];
  log(`sweep ${keeps.join(", ")}: ${SETTLE_MS / 1000} s to settle, ${MEASURE_MS / 1000} s measured each`);
  for (const keep of keeps) {
    const count = await useKeep(keep);
    await sleep(SETTLE_MS);
    restartTiming();
    await sleep(MEASURE_MS);
    const { fps, median } = timing();
    const row = {
      keep,
      count,
      fps: fps ?? 0,
      tier: tier(fps ?? 0),
      landmarker: median.landmarker ?? null,
      draw: median.draw ?? null,
    };
    results.push(row);
    log(`sweep ${describe(row)}`);
  }
  log("sweep done");
  for (const row of results) log(`  ${describe(row)}`);
  return results;
}

function describe({ keep, count, fps, tier, landmarker, draw }) {
  const ms = (v) => (v === null ? "-" : v.toFixed(1));
  return `keep ${keep} ${count} splats: ${fps.toFixed(1)} fps, tier ${tier}, landmarker ${ms(landmarker)} ms, draw ${ms(draw)} ms`;
}
