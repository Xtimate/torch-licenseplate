<script lang="ts">
    import { fade } from "svelte/transition";
    import { onDestroy } from "svelte";

    // ───────────────────────────────────────────────────────────
    // Config
    // ───────────────────────────────────────────────────────────
    const API_BASE = import.meta.env.VITE_API_BASE;
    const WS_BASE = import.meta.env.VITE_API_BASE.replace("http", "ws");

    type Mode =
        | "pipeline"
        | "detect"
        | "recognize"
        | "video"
        | "webcam"
        | "history"
        | "stats"
        | "watchlist";

    interface PlateResult {
        text: string;
        country?: string | null;
        confidence: number;
        conf: number;
        valid_format?: boolean;
        chars?: number[];
        latency_ms?: number;
        error?: string;
    }

    interface HistoryItem {
        text: string;
        country?: string | null;
        confidence: number;
        ts: string;
        source: Mode;
    }

    const MODES: { id: Mode; label: string; hint: string }[] = [
        { id: "pipeline", label: "Pipeline", hint: "Detect + Recognize" },
        { id: "detect", label: "Detect", hint: "Bounding boxes only" },
        { id: "recognize", label: "Recognize", hint: "Cropped plate → text" },
        { id: "video", label: "Video Scan", hint: "Unique plates per clip" },
        { id: "webcam", label: "Live Webcam", hint: "Real-time over WS" },
        { id: "history", label: "History", hint: "All spotted plates" },
        { id: "stats", label: "Stats", hint: "Analytics + heatmap" },
        { id: "watchlist", label: "Watchlist", hint: "Flag specific plates" },
    ];

    const COUNTRIES: Record<string, string> = {
        NL: "Netherlands",
        DE: "Germany",
        FR: "France",
        BE: "Belgium",
    };

    // ───────────────────────────────────────────────────────────
    // State
    // ───────────────────────────────────────────────────────────
    let activeTab: Mode = $state("pipeline");
    let batchMode = $state(false);
    let batchFiles: File[] = $state([]);
    let batchResult: any = $state(null);
    let pipelineFile: File | null = $state(null);
    let detectFile: File | null = $state(null);
    let recognizeFile: File | null = $state(null);
    let videoFile: File | null = $state(null);
    let pipelineResult: PlateResult[] | null = $state(null);
    let detectResult: any = $state(null);
    let recognizeResult: PlateResult | null = $state(null);
    let videoResult: PlateResult[] | null = $state(null);
    let historyResult: any = $state(null);
    let statsResult: any = $state(null);
    let watchlistResult: any[] = $state([]);
    let watchlistInput = $state("");
    let watchlistNotes = $state("");
    let loading = $state(false);
    let lastLatency: number | null = $state(null);
    let previewUrl: string | null = $state(null);
    let history: HistoryItem[] = $state([]);
    let videoEl: HTMLVideoElement | null = $state(null);
    let stream: MediaStream | null = $state(null);
    let ws: WebSocket | null = null;
    let webcamActive = $state(false);
    let webcamResults: PlateResult[] = $state([]);
    let frameInterval: ReturnType<typeof setInterval> | null = null;

    // ───────────────────────────────────────────────────────────
    // Helpers
    // ───────────────────────────────────────────────────────────
    function nowTs() {
        const d = new Date();
        return [d.getHours(), d.getMinutes(), d.getSeconds()]
            .map((n) => String(n).padStart(2, "0"))
            .join(":");
    }

    function pushHistory(plates: PlateResult | PlateResult[], source: Mode) {
        const list = Array.isArray(plates) ? plates : [plates];
        for (const p of list) {
            if (!p || !p.text) continue;
            history = [
                {
                    text: p.text,
                    country: p.country,
                    confidence: p.confidence,
                    ts: nowTs(),
                    source,
                },
                ...history,
            ].slice(0, 30);
        }
    }

    async function post(endpoint: string, file: File | null) {
        if (!file) return null;
        loading = true;
        const t0 = performance.now();
        const form = new FormData();
        form.append("file", file);
        try {
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: "POST",
                body: form,
            });
            const data = await res.json();
            lastLatency = Math.round(performance.now() - t0);
            return data;
        } catch (e: any) {
            return { error: e.message };
        } finally {
            loading = false;
        }
    }

    async function runPipeline() {
        pipelineResult = await post("/pipeline", pipelineFile);
        if (Array.isArray(pipelineResult) && pipelineResult.length > 0) {
            pushHistory(pipelineResult, "pipeline");
            if (pipelineFile) await drawOverlay(pipelineFile, pipelineResult);
    }
    async function runDetect() {
        detectResult = await post("/detect", detectFile);
    }
    async function runRecognize() {
        recognizeResult = await post("/recognize", recognizeFile);
        if (recognizeResult && !recognizeResult.error)
            pushHistory(recognizeResult, "recognize");
    }
    async function runVideo() {
        videoResult = await post("/video", videoFile);
        if (Array.isArray(videoResult)) pushHistory(videoResult, "video");
    }

    async function runBatch() {
        if (!batchFiles.length) return;
        loading = true;
        const t0 = performance.now();
        const form = new FormData();
        for (const file of batchFiles) form.append("files", file);
        try {
            const res = await fetch(`${API_BASE}/pipeline/batch`, {
                method: "POST",
                body: form,
            });
            batchResult = await res.json();
            lastLatency = Math.round(performance.now() - t0);
            if (Array.isArray(batchResult)) {
                for (const g of batchResult) pushHistory(g.plates, "pipeline");
            }
        } catch (e: any) {
            batchResult = { error: e.message };
        } finally {
            loading = false;
        }
    }

    async function startWebcam() {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoEl) videoEl.srcObject = stream;
        webcamActive = true;
        webcamResults = [];

        ws = new WebSocket(`${WS_BASE}/webcam`);
        ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.length > 0) {
                for (const det of data) {
                    if (!webcamResults.find((r) => r.text === det.text)) {
                        webcamResults = [...webcamResults, det];
                        pushHistory(det, "webcam");
                    }
                }
            }
        };

        frameInterval = setInterval(() => {
            if (!videoEl || !ws || ws.readyState !== WebSocket.OPEN) return;
            const canvas = document.createElement("canvas");
            canvas.width = videoEl.videoWidth;
            canvas.height = videoEl.videoHeight;
            canvas.getContext("2d")?.drawImage(videoEl, 0, 0);
            canvas.toBlob(
                (blob) => {
                    if (blob && ws?.readyState === WebSocket.OPEN)
                        ws.send(blob);
                },
                "image/jpeg",
                0.8,
            );
        }, 500);
    }

    function stopWebcam() {
        if (frameInterval) clearInterval(frameInterval);
        ws?.close();
        stream?.getTracks().forEach((t) => t.stop());
        stream = null;
        webcamActive = false;
    }

    onDestroy(() => stopWebcam());

    function setTab(tab: Mode) {
        if (webcamActive) stopWebcam();
        activeTab = tab;
        previewUrl = null;
    }

    function handleFile(e: Event, which: Mode) {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;
        if (which === "pipeline") pipelineFile = file;
        if (which === "detect") detectFile = file;
        if (which === "recognize") recognizeFile = file;
        if (which === "video") videoFile = file;
        if (which !== "video") previewUrl = URL.createObjectURL(file);
    }

    async function loadHistory() {
        const res = await fetch(`${API_BASE}/history?limit=50`);
        historyResult = await res.json();
    }

    async function loadStats() {
        const res = await fetch(`${API_BASE}/stats`);
        statsResult = await res.json();
    }

    async function loadWatchlist() {
        const res = await fetch(`${API_BASE}/watchlist`);
        watchlistResult = await res.json();
    }

    async function addToWatchlist() {
        if (!watchlistInput.trim()) return;
        await fetch(`${API_BASE}/watchlist`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: watchlistInput.trim().toUpperCase(),
                notes: watchlistNotes || null,
            }),
        });
        watchlistInput = "";
        watchlistNotes = "";
        await loadWatchlist();
    }

    async function removeFromWatchlist(text: string) {
        await fetch(`${API_BASE}/watchlist/${text}`, { method: "DELETE" });
        await loadWatchlist();
    }

    async function drawOverlay(file: File, plates: PlateResult[]) {
        const img = new Image();
        img.src = URL.createObjectURL(file);
        await new Promise((res) => (img.onload = res));

        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(img, 0, 0);

        for (const p of plates) {
            const { x1, y1, x2, y2 } = p as any;
            if (x1 == null) continue;

            const w = x2 - x1;
            const h = y2 - y1;

            ctx.strokeStyle = "#e8c84a";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, w, h);
            ctx.fillStyle = "rgba(232, 200, 74, 0.08)";
            ctx.fillText(x1, y1, w, h);

            const label = "${p.text} ${Math.round(p.confidence * 100)}%";
            ctx.font = "bold 18 px DM Mono, monospace";
            const textWidth = ctx.measureText(label).width;
            const labelX = x1;
            const labelY = y1 - 8;

            ctx.fillStyle = "#e8c84a";
            ctx.fillRect(labelX, labelY - 20, textWidth + 12, 24);
            ctx.fillStyle = "0a0a0a";
            ctx.fillText(label, labelX + 6, labelY - 2);
        }

        previewUrl = canvas.toDataURL("image/jpeg", 0.92);
    }
    // ───────────────────────────────────────────────────────────
    // Derived
    // ───────────────────────────────────────────────────────────
    const ACCENT = "#e8c84a";
</script>

<svelte:head>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link
        href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500;600;700&display=swap"
        rel="stylesheet"
    />
</svelte:head>

<div class="root">
    <div class="accent-line"></div>

    <div class="shell">
        <!-- ─── Sidebar ─────────────────────────────────────── -->
        <aside class="sidebar">
            <div class="brand">
                <div class="brand-mark">
                    <span class="bm-tl"></span>
                    <span class="bm-br"></span>
                </div>
                <div>
                    <div class="brand-title">Spotter</div>
                    <div class="brand-sub">v0.4 · ALPR</div>
                </div>
            </div>

            <div class="modes">
                <div class="section-label">Modes</div>
                {#each MODES as m, idx}
                    <button
                        class="mode-btn"
                        class:active={activeTab === m.id}
                        onclick={() => setTab(m.id)}
                    >
                        <span class="mode-num">0{idx + 1}</span>
                        <div class="mode-text">
                            <div class="mode-label">{m.label}</div>
                            <div class="mode-hint">{m.hint}</div>
                        </div>
                    </button>
                {/each}
            </div>

            <div class="history">
                <div class="history-head">
                    <span class="section-label">Recent</span>
                    <span class="muted2">{history.length}</span>
                </div>
                <div class="history-list">
                    {#if history.length === 0}
                        <div class="muted2 tiny pad">No recognitions yet</div>
                    {:else}
                        {#each history.slice(0, 6) as h}
                            <div class="history-row">
                                <span
                                    class="hist-text"
                                    class:invalid={!h.country}>{h.text}</span
                                >
                                <span class="muted2 tiny">{h.ts}</span>
                            </div>
                        {/each}
                    {/if}
                </div>
            </div>

            <div class="status-bar">
                <div class="status-left">
                    <span class="dot pulse"></span>
                    <span class="muted tiny">API · {lastLatency ?? "—"}ms</span>
                </div>
                <span class="muted2 tiny">NL · DE · FR</span>
            </div>
        </aside>

        <!-- ─── Main work area ──────────────────────────────── -->
        <main class="work">
            {#key activeTab}
                <div
                    in:fade={{ duration: 120, delay: 100 }}
                    out:fade={{ duration: 100 }}
                    class="work-inner"
                >
                    <!-- ─── Pipeline ───────────────────────────── -->
                    {#if activeTab === "pipeline"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 01</span>
                                <span class="muted2 tiny">POST</span>
                                <span class="muted tiny mono"
                                    >/pipeline{batchMode ? "/batch" : ""}</span
                                >
                            </div>
                            <h2 class="mode-title">
                                {batchMode ? "Batch Pipeline" : "Pipeline"}
                            </h2>
                            <p class="mode-desc">
                                {batchMode
                                    ? "Process multiple images at once."
                                    : "Detect and recognize all plates in one shot. Returns boxes, plate text, per-character confidence and country validation."}
                            </p>
                        </header>

                        <div class="pill-row">
                            <button
                                class="pill"
                                class:on={!batchMode}
                                onclick={() => {
                                    batchMode = false;
                                    batchResult = null;
                                }}>Single image</button
                            >
                            <button
                                class="pill"
                                class:on={batchMode}
                                onclick={() => {
                                    batchMode = true;
                                    pipelineResult = null;
                                }}>Batch</button
                            >
                        </div>

                        {#if !batchMode}
                            <label class="dropzone">
                                <input
                                    type="file"
                                    accept="image/*"
                                    onchange={(e) => handleFile(e, "pipeline")}
                                />
                                {#if previewUrl}
                                    <img
                                        src={previewUrl}
                                        alt="preview"
                                        class="preview"
                                    />
                                {:else}
                                    <div class="dz-inner">
                                        <span class="dz-plus">+</span>
                                        <div>
                                            <div class="dz-label">
                                                Drop scene photo
                                            </div>
                                            <div class="muted2 tiny">
                                                JPG / PNG · up to 10MB ·
                                                multi-vehicle OK
                                            </div>
                                        </div>
                                    </div>
                                {/if}
                                {#if pipelineFile}<p class="muted tiny mt8">
                                        ✓ {pipelineFile.name}
                                    </p>{/if}
                            </label>

                            <div class="action-row">
                                <button
                                    class="cta"
                                    onclick={runPipeline}
                                    disabled={!pipelineFile || loading}
                                >
                                    {#if loading}<span class="blink">▸</span> Processing…{:else}Run
                                        pipeline{/if}
                                </button>
                                <div class="latency-card">
                                    <div class="muted2 tiny lt">
                                        Last latency
                                    </div>
                                    <div class="latency-val">
                                        <span class="accent"
                                            >{lastLatency ?? "—"}</span
                                        ><span class="muted tiny ml4">ms</span>
                                    </div>
                                </div>
                            </div>

                            {#if pipelineResult && Array.isArray(pipelineResult) && pipelineResult.length > 0}
                                {@const first = pipelineResult[0]}
                                <div class="result-head">
                                    <span class="rh-title">▸ Detections</span>
                                    <span class="muted tiny"
                                        >{pipelineResult.length} found · {lastLatency}ms
                                        total</span
                                    >
                                </div>
                                <div class="result-list">
                                    {#each pipelineResult as p}
                                        {@render plateRow(p)}
                                    {/each}
                                </div>
                                {#if first.chars && first.chars.length}
                                    <div class="card pad-lg">
                                        <div class="row-between mb12">
                                            <span class="section-label"
                                                >Char confidence · {first.text}</span
                                            >
                                            <span class="muted2 tiny"
                                                >min {Math.round(
                                                    Math.min(...first.chars) *
                                                        100,
                                                )}% · max {Math.round(
                                                    Math.max(...first.chars) *
                                                        100,
                                                )}%</span
                                            >
                                        </div>
                                        {@render charConfidence(
                                            first.text,
                                            first.chars,
                                        )}
                                    </div>
                                {/if}
                            {:else if pipelineResult && Array.isArray(pipelineResult)}
                                <div class="empty">No plates detected.</div>
                            {/if}
                        {:else}
                            <label class="dropzone">
                                <input
                                    type="file"
                                    accept="image/*"
                                    multiple
                                    onchange={(e) => {
                                        batchFiles = Array.from(
                                            (e.target as HTMLInputElement)
                                                .files ?? [],
                                        );
                                    }}
                                />
                                <div class="dz-inner">
                                    <span class="dz-plus">+</span>
                                    <div>
                                        <div class="dz-label">
                                            {batchFiles.length
                                                ? `${batchFiles.length} files queued`
                                                : "Select multiple images"}
                                        </div>
                                        <div class="muted2 tiny">
                                            Up to 32 images per batch · runs in
                                            parallel
                                        </div>
                                    </div>
                                </div>
                            </label>

                            <div class="action-row">
                                <button
                                    class="cta"
                                    onclick={runBatch}
                                    disabled={!batchFiles.length || loading}
                                >
                                    {#if loading}<span class="blink">▸</span> Processing…{:else}Run
                                        batch{/if}
                                </button>
                                <div class="latency-card">
                                    <div class="muted2 tiny lt">
                                        Last latency
                                    </div>
                                    <div class="latency-val">
                                        <span class="accent"
                                            >{lastLatency ?? "—"}</span
                                        ><span class="muted tiny ml4">ms</span>
                                    </div>
                                </div>
                            </div>

                            {#if batchResult && Array.isArray(batchResult)}
                                <div class="batch-list">
                                    {#each batchResult as group, gi}
                                        <div class="card pad">
                                            <div class="row-between mb12">
                                                <span class="mode-meta-num"
                                                    >[{gi + 1}] image {group.image_index +
                                                        1}</span
                                                >
                                                <span class="muted tiny"
                                                    >{group.plates.length} plates</span
                                                >
                                            </div>
                                            {#if group.plates.length > 0}
                                                <div class="result-list">
                                                    {#each group.plates as p}
                                                        {@render plateRow(p)}
                                                    {/each}
                                                </div>
                                            {:else}
                                                <p class="muted tiny">
                                                    No plates detected.
                                                </p>
                                            {/if}
                                        </div>
                                    {/each}
                                </div>
                            {/if}
                        {/if}

                        <!-- ─── Detect ─────────────────────────────── -->
                    {:else if activeTab === "detect"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 02</span>
                                <span class="muted2 tiny">POST</span>
                                <span class="muted tiny mono">/detect</span>
                            </div>
                            <h2 class="mode-title">Detect</h2>
                            <p class="mode-desc">
                                Returns bounding boxes only — useful for
                                crop-and-cache pipelines or when you want to run
                                your own OCR downstream.
                            </p>
                        </header>

                        <label class="dropzone">
                            <input
                                type="file"
                                accept="image/*"
                                onchange={(e) => handleFile(e, "detect")}
                            />
                            {#if previewUrl}
                                <img
                                    src={previewUrl}
                                    alt="preview"
                                    class="preview"
                                />
                            {:else}
                                <div class="dz-inner">
                                    <span class="dz-plus">+</span>
                                    <div>
                                        <div class="dz-label">Drop image</div>
                                        <div class="muted2 tiny">
                                            Returns box coordinates + detection
                                            confidence
                                        </div>
                                    </div>
                                </div>
                            {/if}
                            {#if detectFile}<p class="muted tiny mt8">
                                    ✓ {detectFile.name}
                                </p>{/if}
                        </label>

                        <div class="action-row">
                            <button
                                class="cta"
                                onclick={runDetect}
                                disabled={!detectFile || loading}
                            >
                                {#if loading}<span class="blink">▸</span> Detecting…{:else}Detect{/if}
                            </button>
                            <div class="latency-card">
                                <div class="muted2 tiny lt">Last latency</div>
                                <div class="latency-val">
                                    <span class="accent"
                                        >{lastLatency ?? "—"}</span
                                    ><span class="muted tiny ml4">ms</span>
                                </div>
                            </div>
                        </div>

                        {#if detectResult}
                            <div class="card">
                                <div class="result-head inline-head">
                                    <span class="rh-title">▸ JSON response</span
                                    >
                                    <span class="muted tiny"
                                        >{lastLatency}ms</span
                                    >
                                </div>
                                <pre class="json">{JSON.stringify(
                                        detectResult,
                                        null,
                                        2,
                                    )}</pre>
                            </div>
                        {/if}

                        <!-- ─── Recognize ──────────────────────────── -->
                    {:else if activeTab === "recognize"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 03</span>
                                <span class="muted2 tiny">POST</span>
                                <span class="muted tiny mono">/recognize</span>
                            </div>
                            <h2 class="mode-title">Recognize</h2>
                            <p class="mode-desc">
                                Skip detection — feed a pre-cropped plate image.
                                Useful when you've already run detection
                                elsewhere.
                            </p>
                        </header>

                        <label class="dropzone">
                            <input
                                type="file"
                                accept="image/*"
                                onchange={(e) => handleFile(e, "recognize")}
                            />
                            {#if previewUrl}
                                <img
                                    src={previewUrl}
                                    alt="preview"
                                    class="preview"
                                />
                            {:else}
                                <div class="dz-inner">
                                    <span class="dz-plus">+</span>
                                    <div>
                                        <div class="dz-label">
                                            Drop cropped plate
                                        </div>
                                        <div class="muted2 tiny">
                                            Pre-cropped image — no detection
                                            step
                                        </div>
                                    </div>
                                </div>
                            {/if}
                            {#if recognizeFile}<p class="muted tiny mt8">
                                    ✓ {recognizeFile.name}
                                </p>{/if}
                        </label>

                        <div class="action-row">
                            <button
                                class="cta"
                                onclick={runRecognize}
                                disabled={!recognizeFile || loading}
                            >
                                {#if loading}<span class="blink">▸</span> Recognizing…{:else}Recognize{/if}
                            </button>
                            <div class="latency-card">
                                <div class="muted2 tiny lt">Last latency</div>
                                <div class="latency-val">
                                    <span class="accent"
                                        >{lastLatency ?? "—"}</span
                                    ><span class="muted tiny ml4">ms</span>
                                </div>
                            </div>
                        </div>

                        {#if recognizeResult}
                            <div class="card pad-lg">
                                <div class="row-between mb12">
                                    <span class="section-label"
                                        >Per-character readout</span
                                    >
                                    <div class="pill-readout">
                                        <div class="pr-item">
                                            <span class="muted2 tiny lt"
                                                >Country</span
                                            >
                                            <span class="pr-val"
                                                >{recognizeResult.country
                                                    ? (COUNTRIES[
                                                          recognizeResult
                                                              .country
                                                      ] ??
                                                      recognizeResult.country)
                                                    : "—"}</span
                                            >
                                        </div>
                                        <div class="pr-item">
                                            <span class="muted2 tiny lt"
                                                >Format</span
                                            >
                                            <span
                                                class="pr-val"
                                                class:ok={recognizeResult.valid_format}
                                                class:bad={!recognizeResult.valid_format}
                                                >{recognizeResult.valid_format
                                                    ? "✓ Valid"
                                                    : "✗ Invalid"}</span
                                            >
                                        </div>
                                        <div class="pr-item">
                                            <span class="muted2 tiny lt"
                                                >Latency</span
                                            >
                                            <span class="pr-val"
                                                >{lastLatency}ms</span
                                            >
                                        </div>
                                    </div>
                                </div>
                                {#if recognizeResult.chars && recognizeResult.chars.length}
                                    {@render charConfidence(
                                        recognizeResult.text,
                                        recognizeResult.chars,
                                    )}
                                {:else}
                                    <div class="big-plate">
                                        {@render euPlate(
                                            recognizeResult.text,
                                            recognizeResult.country ?? "FR",
                                        )}
                                    </div>
                                {/if}
                            </div>
                        {/if}

                        <!-- ─── Video Scan ─────────────────────────── -->
                    {:else if activeTab === "video"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 04</span>
                                <span class="muted2 tiny">POST</span>
                                <span class="muted tiny mono">/video</span>
                            </div>
                            <h2 class="mode-title">Video Scan</h2>
                            <p class="mode-desc">
                                Scan a clip frame-by-frame. Fuzzy dedupe
                                collapses re-reads of the same plate across
                                frames.
                            </p>
                        </header>

                        <label class="dropzone">
                            <input
                                type="file"
                                accept="video/*"
                                onchange={(e) => handleFile(e, "video")}
                            />
                            <div class="dz-inner">
                                <span class="dz-plus">▶</span>
                                <div>
                                    <div class="dz-label">
                                        {videoFile
                                            ? videoFile.name
                                            : "Drop video clip"}
                                    </div>
                                    <div class="muted2 tiny">
                                        MP4 / MOV · up to 100MB · processed at 4
                                        fps
                                    </div>
                                </div>
                            </div>
                        </label>

                        <div class="action-row">
                            <button
                                class="cta"
                                onclick={runVideo}
                                disabled={!videoFile || loading}
                            >
                                {#if loading}<span class="blink">▸</span> Scanning…{:else}Scan
                                    video{/if}
                            </button>
                            <div class="latency-card">
                                <div class="muted2 tiny lt">Last latency</div>
                                <div class="latency-val">
                                    <span class="accent"
                                        >{lastLatency ?? "—"}</span
                                    ><span class="muted tiny ml4">ms</span>
                                </div>
                            </div>
                        </div>

                        {#if videoResult && Array.isArray(videoResult) && videoResult.length > 0}
                            <div class="result-head">
                                <span class="rh-title">▸ Unique plates</span>
                                <span class="muted tiny"
                                    >{videoResult.length} found · {lastLatency}ms
                                    total</span
                                >
                            </div>
                            <div class="grid-2">
                                {#each videoResult as p}
                                    <div class="card pad row-between">
                                        {@render euPlate(
                                            p.text,
                                            p.country ?? "FR",
                                            "sm",
                                        )}
                                        <div class="col-end">
                                            <span class="accent bold"
                                                >{Math.round(
                                                    p.confidence * 100,
                                                )}%</span
                                            >
                                            {#if p.valid_format}
                                                <span class="muted2 tiny"
                                                    >{COUNTRIES[
                                                        p.country ?? ""
                                                    ] ?? p.country}</span
                                                >
                                            {:else}
                                                <span class="bad tiny"
                                                    >unknown format</span
                                                >
                                            {/if}
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        {:else if videoResult}
                            <div class="empty">No plates found.</div>
                        {/if}

                        <!-- ─── Webcam ─────────────────────────────── -->
                    {:else if activeTab === "webcam"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 05</span>
                                <span class="muted2 tiny">WS</span>
                                <span class="muted tiny mono">/webcam</span>
                            </div>
                            <h2 class="mode-title">Live Webcam</h2>
                            <p class="mode-desc">
                                Stream camera frames over WebSocket to the
                                FastAPI backend. Plates collected as they enter
                                frame.
                            </p>
                        </header>

                        <div class="webcam-grid">
                            <div class="video-card">
                                <video
                                    bind:this={videoEl}
                                    autoplay
                                    playsinline
                                    class="video-el"
                                    class:hidden={!webcamActive}
                                ></video>
                                {#if !webcamActive}
                                    <div class="cam-off">
                                        <div class="muted2 tiny lt">
                                            Camera off
                                        </div>
                                        <div class="muted2 tiny mt8">
                                            Press start to begin streaming
                                        </div>
                                    </div>
                                {:else}
                                    <div class="hud">
                                        <div class="hud-tl">
                                            <span class="dot red pulse"
                                            ></span><span class="tiny accent lt"
                                                >LIVE</span
                                            >
                                        </div>
                                        <div class="hud-tr">
                                            <span class="tiny accent lt"
                                                >WS · {lastLatency ??
                                                    84}ms</span
                                            >
                                        </div>
                                        <div class="hud-bl">
                                            <span class="tiny muted lt"
                                                >SPOTTED · {webcamResults.length}</span
                                            >
                                        </div>
                                        <span class="bracket tl"></span>
                                        <span class="bracket tr"></span>
                                        <span class="bracket bl"></span>
                                        <span class="bracket br"></span>
                                    </div>
                                {/if}
                            </div>

                            <div class="card log-card">
                                <div class="log-head">
                                    <span class="section-label">Live log</span>
                                    <span class="accent tiny"
                                        >{webcamResults.length} unique</span
                                    >
                                </div>
                                <div class="log-list">
                                    {#if webcamResults.length === 0}
                                        <div class="muted2 tiny pad center">
                                            Waiting for plates…
                                        </div>
                                    {:else}
                                        {#each [...webcamResults].reverse() as p}
                                            <div class="log-row">
                                                <div class="row-between">
                                                    <span class="log-text"
                                                        >{p.text}</span
                                                    >
                                                    <span class="muted2 tiny"
                                                        >{nowTs()}</span
                                                    >
                                                </div>
                                                <div class="row-between sm">
                                                    <span class="muted2 tiny"
                                                        >{p.country
                                                            ? (COUNTRIES[
                                                                  p.country
                                                              ] ?? p.country)
                                                            : "?"}</span
                                                    >
                                                    <span class="muted2 tiny"
                                                        >{Math.round(
                                                            p.confidence * 100,
                                                        )}%</span
                                                    >
                                                </div>
                                            </div>
                                        {/each}
                                    {/if}
                                </div>
                            </div>
                        </div>

                        <button
                            class="cta"
                            class:stop={webcamActive}
                            onclick={webcamActive ? stopWebcam : startWebcam}
                        >
                            {webcamActive ? "■ Stop camera" : "▸ Start camera"}
                        </button>

                        <!-- ─── History ────────────────────────────── -->
                    {:else if activeTab === "history"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 06</span>
                                <span class="muted2 tiny">GET</span>
                                <span class="muted tiny mono">/history</span>
                            </div>
                            <h2 class="mode-title">History</h2>
                            <p class="mode-desc">
                                All plates spotted across all sources, newest
                                first.
                            </p>
                        </header>

                        <button class="cta" onclick={loadHistory}
                            >Load history</button
                        >

                        {#if historyResult}
                            <div class="card">
                                {#if historyResult.length === 0}
                                    <div class="empty">
                                        No plates in history yet.
                                    </div>
                                {:else}
                                    <div class="result-list" style="gap: 0;">
                                        {#each historyResult as p}
                                            <div class="history-entry">
                                                {@render euPlate(
                                                    p.text,
                                                    p.country ?? "FR",
                                                    "sm",
                                                )}
                                                <div class="pr-meta">
                                                    {#if p.valid_format}
                                                        <span class="badge ok"
                                                            >✓ {COUNTRIES[
                                                                p.country ?? ""
                                                            ] ??
                                                                p.country}</span
                                                        >
                                                    {:else}
                                                        <span class="badge bad"
                                                            >✗ Invalid</span
                                                        >
                                                    {/if}
                                                    <span class="muted tiny"
                                                        >{p.source}</span
                                                    >
                                                    <span class="muted tiny"
                                                        >{p.confidence
                                                            ? Math.round(
                                                                  p.confidence *
                                                                      100,
                                                              ) + "%"
                                                            : "—"}</span
                                                    >
                                                </div>
                                                <span class="muted2 tiny"
                                                    >{p.timestamp}</span
                                                >
                                            </div>
                                        {/each}
                                    </div>
                                {/if}
                            </div>
                        {/if}

                        <!-- ─── Stats ──────────────────────────────── -->
                    {:else if activeTab === "stats"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 07</span>
                                <span class="muted2 tiny">GET</span>
                                <span class="muted tiny mono">/stats</span>
                            </div>
                            <h2 class="mode-title">Stats</h2>
                            <p class="mode-desc">
                                Aggregate analytics across all spotted plates.
                            </p>
                        </header>

                        <button class="cta" onclick={loadStats}
                            >Load stats</button
                        >

                        {#if statsResult}
                            <div class="grid-2">
                                <div class="card pad">
                                    <div class="section-label mb12">
                                        By country
                                    </div>
                                    {#each statsResult.by_country as c}
                                        <div class="row-between mt8">
                                            <span class="muted"
                                                >{COUNTRIES[c.country ?? ""] ??
                                                    c.country ??
                                                    "Unknown"}</span
                                            >
                                            <span class="accent bold"
                                                >{c.count}</span
                                            >
                                        </div>
                                    {/each}
                                </div>
                                <div class="card pad">
                                    <div class="section-label mb12">
                                        By source
                                    </div>
                                    {#each statsResult.by_source as s}
                                        <div class="row-between mt8">
                                            <span class="muted">{s.source}</span
                                            >
                                            <span class="accent bold"
                                                >{s.count}</span
                                            >
                                        </div>
                                    {/each}
                                </div>
                            </div>

                            <div class="card pad">
                                <div class="section-label mb12">
                                    Top plates · {statsResult.total} total
                                </div>
                                {#each statsResult.top_plates as p}
                                    <div class="row-between mt8">
                                        <span class="accent bold mono"
                                            >{p.text}</span
                                        >
                                        <div
                                            style="display:flex; gap:12px; align-items:center;"
                                        >
                                            <span class="muted tiny"
                                                >{COUNTRIES[p.country ?? ""] ??
                                                    p.country ??
                                                    "?"}</span
                                            >
                                            <span class="muted tiny"
                                                >{p.count}x</span
                                            >
                                        </div>
                                    </div>
                                {/each}
                            </div>

                            {#if statsResult.by_hour.length > 0}
                                <div class="card pad">
                                    <div class="section-label mb12">
                                        Sightings by hour
                                    </div>
                                    <div class="cc-row">
                                        {#each statsResult.by_hour as h}
                                            {@const maxCount = Math.max(
                                                ...statsResult.by_hour.map(
                                                    (x: any) => x.count,
                                                ),
                                            )}
                                            <div class="cc-col">
                                                <span class="cc-pct"
                                                    >{h.count}</span
                                                >
                                                <div class="cc-bar">
                                                    <div
                                                        class="cc-fill"
                                                        style="height: {Math.round(
                                                            (h.count /
                                                                maxCount) *
                                                                100,
                                                        )}%"
                                                    ></div>
                                                </div>
                                                <span class="cc-ch"
                                                    >{h.hour}</span
                                                >
                                            </div>
                                        {/each}
                                    </div>
                                </div>
                            {/if}
                        {/if}

                        <!-- ─── Watchlist ───────────────────────────── -->
                    {:else if activeTab === "watchlist"}
                        <header class="mode-head">
                            <div class="mode-meta">
                                <span class="mode-meta-num">MODE / 08</span>
                                <span class="muted2 tiny"
                                    >GET · POST · DELETE</span
                                >
                                <span class="muted tiny mono">/watchlist</span>
                            </div>
                            <h2 class="mode-title">Watchlist</h2>
                            <p class="mode-desc">
                                Flag specific plates. Any match across pipeline,
                                recognize, video, or webcam will be highlighted.
                            </p>
                        </header>

                        <div class="card pad">
                            <div class="section-label mb12">Add plate</div>
                            <div class="action-row">
                                <input
                                    class="text-input"
                                    placeholder="Plate text e.g. SJ798X"
                                    bind:value={watchlistInput}
                                />
                                <input
                                    class="text-input"
                                    placeholder="Notes (optional)"
                                    bind:value={watchlistNotes}
                                />
                                <button
                                    class="cta"
                                    onclick={addToWatchlist}
                                    style="flex: 0; padding: 14px 20px;"
                                    >Add</button
                                >
                            </div>
                        </div>

                        <button class="cta" onclick={loadWatchlist}
                            >Load watchlist</button
                        >

                        {#if watchlistResult.length > 0}
                            <div class="card">
                                <div class="result-list" style="gap: 0;">
                                    {#each watchlistResult as w}
                                        <div class="history-entry">
                                            {@render euPlate(
                                                w.text,
                                                "NL",
                                                "sm",
                                            )}
                                            <div class="pr-meta">
                                                {#if w.notes}
                                                    <span class="muted tiny"
                                                        >{w.notes}</span
                                                    >
                                                {/if}
                                                <span class="muted2 tiny"
                                                    >{w.added_at}</span
                                                >
                                            </div>
                                            <button
                                                class="badge bad"
                                                style="cursor:pointer;"
                                                onclick={() =>
                                                    removeFromWatchlist(w.text)}
                                                >✕ Remove</button
                                            >
                                        </div>
                                    {/each}
                                </div>
                            </div>
                        {:else if watchlistResult}
                            <div class="empty">Watchlist is empty.</div>
                        {/if}
                    {/if}
                </div>
            {/key}
        </main>
    </div>
</div>

<!-- ─── Snippets ──────────────────────────────────────────────── -->
{#snippet euPlate(text: string, country: string, size: "sm" | "md" = "md")}
    <div class="eu-plate" class:sm={size === "sm"}>
        <div class="eu-band">
            <div class="eu-stars">
                {#each Array(12) as _, i}
                    {@const angle = (i / 12) * Math.PI * 2 - Math.PI / 2}
                    {@const r = 5}
                    <span
                        class="eu-star"
                        style="left: calc(50% + {Math.cos(angle) *
                            r}px); top: calc(50% + {Math.sin(angle) * r}px);"
                    ></span>
                {/each}
            </div>
            <span class="eu-cc">{country}</span>
        </div>
        <span class="eu-text">{text}</span>
    </div>
{/snippet}

{#snippet plateRow(p: PlateResult)}
    <div class="plate-row">
        {@render euPlate(p.text, p.country ?? "FR")}
        <div class="pr-meta">
            {#if p.valid_format}
                <span class="badge ok"
                    >✓ {COUNTRIES[p.country ?? ""] ?? p.country}</span
                >
            {:else}
                <span class="badge bad">✗ Invalid format</span>
            {/if}
            <span class="muted tiny"
                >recog <span class="accent"
                    >{Math.round(p.confidence * 100)}%</span
                ></span
            >
            <span class="muted tiny"
                >det <span class="accent">{Math.round(p.conf * 100)}%</span
                ></span
            >
            {#if p.latency_ms}<span class="muted tiny">{p.latency_ms}ms</span
                >{/if}
        </div>
        <div class="meter-col">
            <div class="meter">
                <div
                    class="meter-fill"
                    style="width: {p.confidence * 100}%"
                ></div>
            </div>
            <span class="muted2 tiny lt">OVERALL</span>
        </div>
    </div>
{/snippet}

{#snippet charConfidence(text: string, chars: number[])}
    {@const charsArr = chars}
    <div class="cc-row">
        {#each text.split("") as ch, i}
            {@const ci = text.slice(0, i).replace(/[-\s]/g, "").length}
            {@const isSep = ch === "-" || ch === " "}
            {@const c = isSep ? null : charsArr[ci]}
            {@const lowConf = c != null && c < 0.85}
            <div class="cc-col">
                <div class="cc-pct" class:lc={lowConf} class:none={c == null}>
                    {c == null ? "—" : Math.round(c * 100)}
                </div>
                <div class="cc-bar">
                    {#if !isSep && c != null}
                        <div
                            class="cc-fill"
                            class:lc={lowConf}
                            style="height: {c * 100}%"
                        ></div>
                    {/if}
                </div>
                <div class="cc-ch" class:sep={isSep}>{ch}</div>
            </div>
        {/each}
    </div>
{/snippet}

<style>
    :global(:root) {
        --bg: #0a0a0a;
        --fg: #ffffff;
        --accent: #e8c84a;
        --mute: rgba(255, 255, 255, 0.4);
        --mute2: rgba(255, 255, 255, 0.25);
        --line: rgba(255, 255, 255, 0.06);
        --line-strong: rgba(255, 255, 255, 0.1);
        --card-bg: rgba(255, 255, 255, 0.025);
        --good: #10b981;
        --bad: #ef4444;
    }

    .root {
        min-height: 100vh;
        background: var(--bg);
        color: var(--fg);
        font-family: "DM Mono", monospace;
        display: flex;
        flex-direction: column;
    }
    .accent-line {
        height: 2px;
        background: var(--accent);
        flex-shrink: 0;
    }
    .shell {
        display: flex;
        flex: 1;
        min-height: 0;
    }

    .sidebar {
        width: 260px;
        flex-shrink: 0;
        border-right: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        position: sticky;
        top: 0;
        height: 100vh;
    }
    .brand {
        padding: 24px 22px 20px;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-mark {
        width: 28px;
        height: 28px;
        position: relative;
        border: 2px solid var(--accent);
        border-radius: 4px;
    }
    .bm-tl,
    .bm-br {
        position: absolute;
        width: 6px;
        height: 6px;
        background: var(--accent);
        border-radius: 1px;
    }
    .bm-tl {
        left: 2px;
        top: 2px;
    }
    .bm-br {
        right: 2px;
        bottom: 2px;
    }
    .brand-title {
        font-family: "Syne", sans-serif;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .brand-sub {
        font-size: 9px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--mute2);
        margin-top: 1px;
    }

    .modes {
        flex: 1;
        padding: 20px 14px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        overflow-y: auto;
    }
    .section-label {
        font-size: 9px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--mute2);
        padding: 0 8px 8px;
    }
    .mode-btn {
        all: unset;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 4px;
        border-left: 2px solid transparent;
        transition:
            background 0.15s,
            border-color 0.15s;
    }
    .mode-btn:hover {
        background: rgba(255, 255, 255, 0.03);
    }
    .mode-btn.active {
        background: rgba(232, 200, 74, 0.08);
        border-left-color: var(--accent);
    }
    .mode-num {
        font-size: 9px;
        color: var(--mute2);
        width: 14px;
    }
    .mode-btn.active .mode-num {
        color: var(--accent);
    }
    .mode-text {
        flex: 1;
    }
    .mode-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .mode-btn.active .mode-label {
        color: var(--accent);
    }
    .mode-hint {
        font-size: 10px;
        margin-top: 2px;
        color: var(--mute2);
    }

    .history {
        border-top: 1px solid var(--line);
        padding: 16px 14px;
    }
    .history-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 8px 8px;
    }
    .history-list {
        display: flex;
        flex-direction: column;
        gap: 1px;
        max-height: 200px;
        overflow: auto;
    }
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 8px;
        border-radius: 3px;
        font-size: 11px;
    }
    .hist-text {
        color: var(--accent);
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .hist-text.invalid {
        color: var(--bad);
    }

    .status-bar {
        border-top: 1px solid var(--line);
        padding: 12px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-shrink: 0;
    }
    .status-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--good);
    }
    .dot.red {
        background: var(--bad);
    }

    .work {
        flex: 1;
        overflow: auto;
        padding: 32px 40px;
        min-width: 0;
    }
    .work-inner {
        max-width: 920px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .mode-head {
        margin-bottom: 4px;
    }
    .mode-meta {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .mode-meta-num {
        font-size: 10px;
        color: var(--accent);
        letter-spacing: 3px;
    }
    .mode-title {
        font-family: "Syne", sans-serif;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }
    .mode-desc {
        font-size: 12px;
        margin-top: 6px;
        max-width: 480px;
        line-height: 1.6;
        color: var(--mute);
    }

    .pill-row {
        display: flex;
        gap: 8px;
    }
    .pill {
        all: unset;
        cursor: pointer;
        padding: 8px 16px;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
        border: 1px solid var(--line-strong);
        border-radius: 3px;
        color: var(--mute);
    }
    .pill.on {
        background: var(--accent);
        color: #0a0a0a;
        border-color: var(--accent);
    }

    .dropzone {
        display: block;
        width: 100%;
        padding: 28px;
        text-align: center;
        background: rgba(255, 255, 255, 0.015);
        border: 1px dashed var(--line-strong);
        border-radius: 6px;
        cursor: pointer;
        transition: border-color 0.15s;
        box-sizing: border-box;
    }
    .dropzone:hover {
        border-color: rgba(232, 200, 74, 0.4);
    }
    .dropzone input {
        display: none;
    }
    .dz-inner {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .dz-plus {
        width: 32px;
        height: 32px;
        border: 1px solid var(--accent);
        border-radius: 3px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--accent);
        font-size: 18px;
        font-weight: 300;
    }
    .dz-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-align: left;
    }
    .preview {
        max-height: 200px;
        margin: 0 auto;
        border-radius: 4px;
        object-fit: contain;
    }

    .action-row {
        display: flex;
        gap: 12px;
        align-items: stretch;
    }
    .cta {
        all: unset;
        cursor: pointer;
        flex: 1;
        padding: 14px 24px;
        background: var(--accent);
        color: #0a0a0a;
        text-align: center;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-radius: 4px;
        transition: background 0.15s;
    }
    .cta:hover:not(:disabled) {
        background: #f0d060;
    }
    .cta:disabled {
        opacity: 0.3;
        cursor: not-allowed;
    }
    .cta.stop {
        background: rgba(239, 68, 68, 0.1);
        color: var(--bad);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .latency-card {
        padding: 0 16px;
        display: flex;
        align-items: center;
        border: 1px solid var(--line);
        border-radius: 4px;
        text-align: right;
    }
    .latency-val {
        font-size: 14px;
        font-weight: 700;
    }

    .result-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: rgba(232, 200, 74, 0.05);
        border-left: 2px solid var(--accent);
    }
    .result-head.inline-head {
        background: transparent;
        border: none;
        padding: 10px 14px;
        border-bottom: 1px solid var(--line);
    }
    .rh-title {
        font-size: 10px;
        color: var(--accent);
        letter-spacing: 2px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .result-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .plate-row {
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 20px;
        padding: 16px 18px;
        background: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 4px;
        align-items: center;
    }
    .history-entry {
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 16px;
        padding: 12px 16px;
        border-bottom: 1px solid var(--line);
        align-items: center;
    }
    .history-entry:last-child {
        border-bottom: none;
    }
    .pr-meta {
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
    }
    .meter-col {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 4px;
    }
    .meter {
        width: 80px;
        height: 4px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 2px;
        overflow: hidden;
    }
    .meter-fill {
        height: 100%;
        background: var(--accent);
    }

    .badge {
        font-size: 9px;
        padding: 3px 8px;
        border-radius: 2px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
        border: 1px solid;
    }
    .badge.ok {
        background: rgba(16, 185, 129, 0.12);
        color: var(--good);
        border-color: rgba(16, 185, 129, 0.3);
    }
    .badge.bad {
        background: rgba(239, 68, 68, 0.1);
        color: var(--bad);
        border-color: rgba(239, 68, 68, 0.3);
    }

    .text-input {
        background: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 8px 12px;
        color: white;
        font-family: "DM Mono", monospace;
        font-size: 12px;
        flex: 1;
    }
    .text-input:focus {
        outline: none;
        border-color: var(--accent);
    }

    .eu-plate {
        display: inline-flex;
        width: 220px;
        height: 56px;
        background: #fff;
        border: 2px solid #111;
        border-radius: 6px;
        overflow: hidden;
        font-family: "DM Mono", monospace;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        flex-shrink: 0;
    }
    .eu-plate.sm {
        width: 140px;
        height: 36px;
    }
    .eu-band {
        width: 30px;
        background: #003399;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0;
        flex-shrink: 0;
    }
    .eu-plate.sm .eu-band {
        width: 20px;
    }
    .eu-stars {
        position: relative;
        width: 14px;
        height: 14px;
    }
    .eu-plate.sm .eu-stars {
        width: 10px;
        height: 10px;
    }
    .eu-star {
        position: absolute;
        width: 1.5px;
        height: 1.5px;
        background: #ffd700;
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }
    .eu-cc {
        color: #ffd700;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .eu-plate.sm .eu-cc {
        font-size: 8px;
    }
    .eu-text {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0a0a0a;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 1.4px;
    }
    .eu-plate.sm .eu-text {
        font-size: 13px;
        letter-spacing: 1px;
    }

    .cc-row {
        display: flex;
        gap: 4px;
        align-items: flex-end;
    }
    .cc-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        min-width: 26px;
    }
    .cc-pct {
        font-size: 10px;
        color: var(--mute);
    }
    .cc-pct.lc {
        color: var(--bad);
    }
    .cc-pct.none {
        color: var(--mute2);
    }
    .cc-bar {
        width: 24px;
        height: 36px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 2px;
        position: relative;
        overflow: hidden;
    }
    .cc-fill {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--accent);
        opacity: 0.85;
    }
    .cc-fill.lc {
        background: var(--bad);
    }
    .cc-ch {
        font-size: 13px;
        font-weight: 700;
    }
    .cc-ch.sep {
        color: var(--mute2);
    }

    .webcam-grid {
        display: grid;
        grid-template-columns: 1fr 320px;
        gap: 16px;
        min-height: 380px;
    }
    .video-card {
        background: #000;
        border: 1px solid var(--line);
        border-radius: 4px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        aspect-ratio: 16/9;
        overflow: hidden;
    }
    .video-el {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .video-el.hidden {
        display: none;
    }
    .cam-off {
        text-align: center;
    }
    .hud {
        position: absolute;
        inset: 0;
        pointer-events: none;
    }
    .hud-tl,
    .hud-tr,
    .hud-bl {
        position: absolute;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hud-tl {
        left: 16px;
        top: 16px;
    }
    .hud-tr {
        right: 16px;
        top: 16px;
    }
    .hud-bl {
        left: 16px;
        bottom: 16px;
    }
    .bracket {
        position: absolute;
        width: 18px;
        height: 18px;
    }
    .bracket.tl {
        left: 12px;
        top: 12px;
        border-top: 1px solid var(--accent);
        border-left: 1px solid var(--accent);
    }
    .bracket.tr {
        right: 12px;
        top: 12px;
        border-top: 1px solid var(--accent);
        border-right: 1px solid var(--accent);
    }
    .bracket.bl {
        left: 12px;
        bottom: 12px;
        border-bottom: 1px solid var(--accent);
        border-left: 1px solid var(--accent);
    }
    .bracket.br {
        right: 12px;
        bottom: 12px;
        border-bottom: 1px solid var(--accent);
        border-right: 1px solid var(--accent);
    }

    .log-card {
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .log-head {
        padding: 10px 14px;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .log-list {
        flex: 1;
        overflow: auto;
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .log-row {
        padding: 10px;
        background: rgba(232, 200, 74, 0.04);
        border-left: 2px solid var(--accent);
        border-radius: 2px;
    }
    .log-text {
        font-size: 13px;
        color: var(--accent);
        font-weight: 700;
        letter-spacing: 1px;
    }

    .grid-2 {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }
    .col-end {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
    }
    .col-end-list {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
    }

    .pill-readout {
        display: flex;
        gap: 14px;
    }
    .pr-item {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
    }
    .pr-val {
        font-size: 12px;
        font-weight: 700;
        color: var(--accent);
    }
    .pr-val.ok {
        color: var(--good);
    }
    .pr-val.bad {
        color: var(--bad);
    }
    .big-plate {
        display: flex;
        justify-content: center;
        padding: 20px 0;
        transform: scale(1.4);
        transform-origin: center;
    }

    .card {
        background: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 4px;
        overflow: hidden;
    }
    .pad {
        padding: 16px;
    }
    .pad-lg {
        padding: 20px;
    }
    .row-between {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .row-between.sm {
        font-size: 9px;
        margin-top: 4px;
    }
    .json {
        margin: 0;
        padding: 18px;
        font-size: 11px;
        line-height: 1.7;
        color: rgba(255, 255, 255, 0.7);
        overflow-x: auto;
        font-family: "DM Mono", monospace;
    }
    .empty {
        padding: 16px;
        font-size: 11px;
        color: var(--mute);
    }

    .accent {
        color: var(--accent);
    }
    .muted {
        color: var(--mute);
    }
    .muted2 {
        color: var(--mute2);
    }
    .tiny {
        font-size: 10px;
    }
    .lt {
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .mono {
        font-family: "DM Mono", monospace;
    }
    .ml4 {
        margin-left: 4px;
    }
    .mt8 {
        margin-top: 8px;
    }
    .mb12 {
        margin-bottom: 12px;
    }
    .bold {
        font-weight: 700;
    }
    .ok {
        color: var(--good);
    }
    .bad {
        color: var(--bad);
    }
    .center {
        text-align: center;
    }

    .pulse {
        animation: spotter-pulse 1.6s infinite;
    }
    @keyframes spotter-pulse {
        0%,
        100% {
            opacity: 1;
        }
        50% {
            opacity: 0.4;
        }
    }
    .blink {
        animation: spotter-blink 1s infinite;
    }
    @keyframes spotter-blink {
        0%,
        49% {
            opacity: 1;
        }
        50%,
        100% {
            opacity: 0;
        }
    }

    .batch-list {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    @media (max-width: 900px) {
        .shell {
            flex-direction: column;
        }
        .sidebar {
            width: 100%;
            height: auto;
            position: static;
            border-right: none;
            border-bottom: 1px solid var(--line);
        }
        .modes {
            flex-direction: row;
            overflow-x: auto;
            padding: 12px;
        }
        .mode-btn {
            flex-shrink: 0;
            min-width: 140px;
        }
        .history,
        .status-bar {
            display: none;
        }
        .work {
            padding: 24px 20px;
        }
        .webcam-grid {
            grid-template-columns: 1fr;
        }
        .grid-2 {
            grid-template-columns: 1fr;
        }
    }
</style>
