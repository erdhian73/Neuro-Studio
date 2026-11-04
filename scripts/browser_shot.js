#!/usr/bin/env node
/**
 * Headless-browser inspector for the Neuro Studio frontend.
 *
 * Uses the Chromium that already ships with playwright-core in
 * frontend/node_modules (no extra install, no network needed) to load the live
 * dev server, capture screenshots, and echo every console message / page error
 * so a broken render is visible from the terminal.
 *
 * Usage
 * -----
 *     node scripts/browser_shot.js                       # default 1440x900
 *     node scripts/browser_shot.js --out /tmp/neuro.png --width 1600 --height 1000
 *     node scripts/browser_shot.js --play              # click Play, wait 2 s
 *     node scripts/browser_shot.js --full              # full-page screenshot
 *
 * Env
 * ---
 *     NEURO_URL   default http://127.0.0.1:3123
 */

const fs = require('node:fs')
const path = require('node:path')

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? fallback : process.argv[i + 1]
}
const flag = (name) => process.argv.includes(`--${name}`)

const {
  chromium,
} = require(path.join(__dirname, '..', 'frontend', 'node_modules', 'playwright'))

const url = process.env.NEURO_URL || 'http://127.0.0.1:3123'
const out = arg('out', '/tmp/neuro-studio.png')
const width = Number(arg('width', 1440))
const height = Number(arg('height', 900))

;(async () => {
  const browser = await chromium.launch({
    // Playwright's bundled headless shell (installed via
    // `npx playwright install chromium-headless-shell`). The full Chromium
    // build segfaults on this OS, so do NOT pass channel: 'chromium'.
    // Sandboxed CI/dev containers usually cannot use the kernel sandbox.
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  })
  const page = await browser.newPage({ viewport: { width, height } })

  const logs = []
  page.on('console', (msg) => logs.push(`[console.${msg.type()}] ${msg.text()}`))
  page.on('pageerror', (err) => logs.push(`[pageerror] ${err.message}`))
  page.on('requestfailed', (req) =>
    logs.push(`[requestfailed] ${req.url()} ${req.failure()?.errorText ?? ''}`)
  )

  const started = Date.now()
  const response = await page.goto(url, { waitUntil: 'networkidle', timeout: 60_000 })
  // Let the <video> report its real width/height so the header renders.
  await page.waitForTimeout(1500)

  const facts = await page.evaluate(() => {
    const video = document.querySelector('video')
    const text = document.body.innerText
    return {
      title: document.title,
      videoSrc: video?.currentSrc || video?.getAttribute('src') || null,
      videoSize: video ? `${video.videoWidth}x${video.videoHeight}` : null,
      videoDuration: video ? Number(video.duration.toFixed(3)) : null,
      videoReadyState: video?.readyState ?? null,
      headerMeta: document.querySelector('.canvas-meta')?.textContent?.trim() ?? null,
      // Player readouts come straight from the DOM, so a stale duration()
      // (00:00 for a 25 s clip) is visible without looking at the pixels.
      player: {
        timecode: document.querySelector('.timecode')?.textContent ?? null,
        duration: document.querySelector('.duration')?.textContent ?? null,
        scrubMax: document.querySelector('.scrub')?.getAttribute('aria-valuemax') ?? null,
      },
      bodyChars: text.length,
      bodyHead: text.slice(0, 240).replace(/\s+/g, ' '),
      clipsInCache: text.includes('DEMO SAMPLES'),

      // --- live wiring checks -------------------------------------------
      // Where the numbers came from: "CACHED RUN" vs "LIVE GPU RUN".
      sourceBadge: document.querySelector('.live-pill')?.textContent?.trim() ?? null,
      // Telemetry bars, read from the DOM: these must change with the
      // playhead if the strip is really driven by the studio payload.
      telemetry: Array.from(document.querySelectorAll('.network')).map((row) => ({
        label: row.querySelector('span')?.textContent?.trim() ?? null,
        value: row.querySelector('strong')?.textContent?.trim() ?? null,
        static: row.classList.contains('network-static'),
      })),
      // Counterfeit-data check: the old hardcoded copy is gone.
      counterfeit: {
        old92: text.includes('94% confidence'),
        oldHook: text.includes('Your brain already knows where to look'),
        old34: text.includes('+34%'),
      },
      verdictSummary: document.querySelector('.verdict-summary')?.textContent?.trim() ?? null,
      insights: Array.from(document.querySelectorAll('.insight')).map((row) => ({
        timestamp: row.querySelector('.insight-seek')?.textContent?.trim() ?? null,
        text: row.querySelector('p')?.textContent?.trim() ?? null,
      })),
      samples: Array.from(document.querySelectorAll('.sample')).map((button) => ({
        label: button.textContent?.trim() ?? null,
        active: button.classList.contains('active'),
      })),
    }
  })

  if (flag('play')) {
    await page.locator('.play-button').click()
    await page.waitForTimeout(2000)
    facts.afterPlay = await page.evaluate(() => ({
      timecode: document.querySelector('.timecode')?.textContent ?? null,
      paused: document.querySelector('video')?.paused,
      telemetry: Array.from(document.querySelectorAll('.network strong')).map(
        (el) => el.textContent?.trim() ?? null,
      ),
    }))
  }

  // `--seek <seconds>`: jump the real <video> element forward and re-read the
  // telemetry bars. If the strip is wired to the studio payload the numbers
  // must differ from the ones captured at t=0.
  const seekTo = arg('seek', null)
  if (seekTo !== null) {
    await page.evaluate((seconds) => {
      const video = document.querySelector('video')
      if (video) {
        video.currentTime = seconds
        video.dispatchEvent(new Event('timeupdate'))
      }
    }, Number(seekTo))
    await page.waitForTimeout(600)
    facts.afterSeek = await page.evaluate(() => ({
      timecode: document.querySelector('.timecode')?.textContent ?? null,
      scrubNow: document.querySelector('.scrub')?.getAttribute('aria-valuenow') ?? null,
      telemetry: Array.from(document.querySelectorAll('.network strong')).map(
        (el) => el.textContent?.trim() ?? null,
      ),
    }))
    await page.screenshot({ path: out.replace(/\.png$/, '-seek.png') })
  }

  fs.mkdirSync(path.dirname(out), { recursive: true })
  await page.screenshot({ path: out, fullPage: flag('full') })
  await browser.close()

  console.log(
    JSON.stringify(
      { url, status: response.status(), elapsedMs: Date.now() - started, out, facts, logs },
      null,
      2
    )
  )
})().catch((err) => {
  console.error(err)
  process.exit(1)
})
