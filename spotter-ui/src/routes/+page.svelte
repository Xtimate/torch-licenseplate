<script lang="ts">
    import { fade } from "svelte/transition";
    import { onDestroy } from "svelte";

    const API_BASE = import.meta.env.VITE_API_BASE;
    const WS_BASE = import.meta.env.VITS_API_BASE.replace("http", "ws");
    let activeTab = $state("pipeline");
    let pipelineFile: File | null = $state(null);
    let detectFile: File | null = $state(null);
    let recognizeFile: File | null = $state(null);
    let videoFile: File | null = $state(null);
    let pipelineResult: any = $state(null);
    let detectResult: any = $state(null);
    let recognizeResult: any = $state(null);
    let videoResult: any = $state(null);
    let loading = $state(false);
    let previewUrl: string | null = $state(null);
    let videoEl: HTMLVideoElement | null = $state(null);
    let stream: MediaStream | null = $state(null);
    let ws: WebSocket | null = null;
    let webcamActive = $state(false);
    let webcamResults: any[] = $state([]);
    let frameInterval: ReturnType<typeof setInterval> | null = null;

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
        clearInterval(frameInterval!);
        ws?.close();
        stream?.getTracks().forEach((t) => t.stop());
        stream = null;
        webcamActive = false;
    }

    onDestroy(() => stopWebcam());
    function setTab(tab: string) {
        if (webcamActive) stopWebcam();
        activeTab = tab;
        previewUrl = null;
    }

    function handleFile(e: Event, which: string) {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;
        if (which === "pipeline") pipelineFile = file;
        if (which === "detect") detectFile = file;
        if (which === "recognize") recognizeFile = file;
        if (which === "video") videoFile = file;
        if (which !== "video") previewUrl = URL.createObjectURL(file);
    }

    async function post(endpoint: string, file: File | null) {
        if (!file) return null;
        loading = true;
        const form = new FormData();
        form.append("file", file);
        try {
            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: "POST",
                body: form,
            });
            return await res.json();
        } catch (e: any) {
            return { error: e.message };
        } finally {
            loading = false;
        }
    }

    async function runPipeline() {
        pipelineResult = await post("/pipeline", pipelineFile);
    }
    async function runDetect() {
        detectResult = await post("/detect", detectFile);
    }
    async function runRecognize() {
        recognizeResult = await post("/recognize", recognizeFile);
    }
    async function runVideo() {
        videoResult = await post("/video", videoFile);
    }
</script>

<svelte:head>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link
        href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap"
        rel="stylesheet"
    />
</svelte:head>

<div class="min-h-screen bg-[#0a0a0a] text-white font-mono">
    <div class="h-0.5 w-full bg-[#e8c84a]"></div>

    <header
        class="px-8 py-5 flex items-center justify-between border-b border-white/5"
    >
        <div>
            <span class="text-[10px] tracking-[4px] text-[#e8c84a] uppercase"
                >License Plate Recognition</span
            >
            <h1
                class="text-2xl font-black tracking-tight mt-0.5"
                style="font-family: 'Syne', sans-serif;"
            >
                Spotter
            </h1>
        </div>
        <div
            class="flex items-center gap-2 text-[11px] text-white/30 tracking-widest uppercase"
        >
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"
            ></span>
            API online
        </div>
    </header>

    <nav class="px-8 flex gap-0 border-b border-white/5">
        {#each ["pipeline", "detect", "recognize", "video", "webcam"] as tab}
            <button
                onclick={() => setTab(tab)}
                class="px-5 py-3.5 text-[11px] tracking-[2px] uppercase transition-all border-b-2 {activeTab ===
                tab
                    ? 'border-[#e8c84a] text-[#e8c84a]'
                    : 'border-transparent text-white/30 hover:text-white/60'}"
            >
                {tab}
            </button>
        {/each}
    </nav>

    <main class="px-8 py-10 max-w-4xl mx-auto">
        {#key activeTab}
            <div
                in:fade={{ duration: 120, delay: 120 }}
                out:fade={{ duration: 120 }}
            >
                {#if activeTab === "pipeline"}
                    <div class="space-y-6">
                        <div>
                            <p
                                class="text-[10px] tracking-[3px] text-white/30 uppercase mb-1"
                            >
                                Mode
                            </p>
                            <h2
                                class="text-xl font-bold"
                                style="font-family: 'Syne', sans-serif;"
                            >
                                Full Pipeline
                            </h2>
                            <p class="text-white/30 text-xs mt-1">
                                Detect plates in an image and recognize all text
                                in one shot.
                            </p>
                        </div>

                        <label
                            class="block border border-dashed border-white/10 rounded-lg p-8 text-center cursor-pointer hover:border-[#e8c84a]/40 transition-colors group"
                        >
                            <input
                                type="file"
                                accept="image/*"
                                class="hidden"
                                onchange={(e) => handleFile(e, "pipeline")}
                            />
                            {#if previewUrl}
                                <img
                                    src={previewUrl}
                                    alt="preview"
                                    class="max-h-48 mx-auto rounded mb-3 object-contain"
                                />
                            {:else}
                                <div
                                    class="text-white/20 group-hover:text-white/40 transition-colors"
                                >
                                    <div class="text-3xl mb-2">+</div>
                                    <div
                                        class="text-[11px] tracking-widest uppercase"
                                    >
                                        Drop image or click
                                    </div>
                                </div>
                            {/if}
                            {#if pipelineFile}<p
                                    class="text-[10px] text-white/30 mt-2"
                                >
                                    {pipelineFile.name}
                                </p>{/if}
                        </label>

                        <button
                            onclick={runPipeline}
                            disabled={!pipelineFile || loading}
                            class="w-full py-3 bg-[#e8c84a] text-[#0a0a0a] text-[11px] tracking-[3px] uppercase font-bold rounded disabled:opacity-30 hover:bg-[#f0d060] transition-colors"
                        >
                            {loading ? "Running..." : "Run Pipeline"}
                        </button>

                        {#if pipelineResult}
                            <div
                                class="border border-white/5 rounded-lg overflow-hidden"
                            >
                                <div
                                    class="px-4 py-2 bg-white/3 border-b border-white/5 text-[10px] tracking-widest text-white/30 uppercase"
                                >
                                    Result
                                </div>
                                {#if Array.isArray(pipelineResult) && pipelineResult.length > 0}
                                    <div class="p-4 space-y-2">
                                        {#each pipelineResult as det}
                                            <div
                                                class="flex items-center justify-between bg-white/3 rounded px-4 py-3"
                                            >
                                                <span
                                                    class="text-[#e8c84a] font-bold tracking-widest text-lg"
                                                    style="font-family: 'Syne', sans-serif;"
                                                    >{det.text}</span
                                                >
                                                <span
                                                    class="text-[10px] text-white/30"
                                                    >conf {(
                                                        det.conf * 100
                                                    ).toFixed(0)}%</span
                                                >
                                            </div>
                                        {/each}
                                    </div>
                                {:else}
                                    <p class="p-4 text-white/30 text-xs">
                                        No plates detected.
                                    </p>
                                {/if}
                            </div>
                        {/if}
                    </div>
                {:else if activeTab === "detect"}
                    <div class="space-y-6">
                        <div>
                            <p
                                class="text-[10px] tracking-[3px] text-white/30 uppercase mb-1"
                            >
                                Mode
                            </p>
                            <h2
                                class="text-xl font-bold"
                                style="font-family: 'Syne', sans-serif;"
                            >
                                Detect Plates
                            </h2>
                            <p class="text-white/30 text-xs mt-1">
                                Returns bounding boxes for all plates found in
                                the image.
                            </p>
                        </div>

                        <label
                            class="block border border-dashed border-white/10 rounded-lg p-8 text-center cursor-pointer hover:border-[#e8c84a]/40 transition-colors group"
                        >
                            <input
                                type="file"
                                accept="image/*"
                                class="hidden"
                                onchange={(e) => handleFile(e, "detect")}
                            />
                            {#if previewUrl}
                                <img
                                    src={previewUrl}
                                    alt="preview"
                                    class="max-h-48 mx-auto rounded mb-3 object-contain"
                                />
                            {:else}
                                <div
                                    class="text-white/20 group-hover:text-white/40 transition-colors"
                                >
                                    <div class="text-3xl mb-2">+</div>
                                    <div
                                        class="text-[11px] tracking-widest uppercase"
                                    >
                                        Drop image or click
                                    </div>
                                </div>
                            {/if}
                            {#if detectFile}<p
                                    class="text-[10px] text-white/30 mt-2"
                                >
                                    {detectFile.name}
                                </p>{/if}
                        </label>

                        <button
                            onclick={runDetect}
                            disabled={!detectFile || loading}
                            class="w-full py-3 bg-[#e8c84a] text-[#0a0a0a] text-[11px] tracking-[3px] uppercase font-bold rounded disabled:opacity-30 hover:bg-[#f0d060] transition-colors"
                        >
                            {loading ? "Detecting..." : "Detect"}
                        </button>

                        {#if detectResult}
                            <div
                                class="border border-white/5 rounded-lg overflow-hidden"
                            >
                                <div
                                    class="px-4 py-2 bg-white/3 border-b border-white/5 text-[10px] tracking-widest text-white/30 uppercase"
                                >
                                    Detections
                                </div>
                                <pre
                                    class="p-4 text-xs text-white/50 overflow-x-auto">{JSON.stringify(
                                        detectResult,
                                        null,
                                        2,
                                    )}</pre>
                            </div>
                        {/if}
                    </div>
                {:else if activeTab === "recognize"}
                    <div class="space-y-6">
                        <div>
                            <p
                                class="text-[10px] tracking-[3px] text-white/30 uppercase mb-1"
                            >
                                Mode
                            </p>
                            <h2
                                class="text-xl font-bold"
                                style="font-family: 'Syne', sans-serif;"
                            >
                                Recognize Plate
                            </h2>
                            <p class="text-white/30 text-xs mt-1">
                                Feed a cropped plate image directly to the
                                recognizer.
                            </p>
                        </div>

                        <label
                            class="block border border-dashed border-white/10 rounded-lg p-8 text-center cursor-pointer hover:border-[#e8c84a]/40 transition-colors group"
                        >
                            <input
                                type="file"
                                accept="image/*"
                                class="hidden"
                                onchange={(e) => handleFile(e, "recognize")}
                            />
                            {#if previewUrl}
                                <img
                                    src={previewUrl}
                                    alt="preview"
                                    class="max-h-48 mx-auto rounded mb-3 object-contain"
                                />
                            {:else}
                                <div
                                    class="text-white/20 group-hover:text-white/40 transition-colors"
                                >
                                    <div class="text-3xl mb-2">+</div>
                                    <div
                                        class="text-[11px] tracking-widest uppercase"
                                    >
                                        Drop cropped plate or click
                                    </div>
                                </div>
                            {/if}
                            {#if recognizeFile}<p
                                    class="text-[10px] text-white/30 mt-2"
                                >
                                    {recognizeFile.name}
                                </p>{/if}
                        </label>

                        <button
                            onclick={runRecognize}
                            disabled={!recognizeFile || loading}
                            class="w-full py-3 bg-[#e8c84a] text-[#0a0a0a] text-[11px] tracking-[3px] uppercase font-bold rounded disabled:opacity-30 hover:bg-[#f0d060] transition-colors"
                        >
                            {loading ? "Recognizing..." : "Recognize"}
                        </button>

                        {#if recognizeResult}
                            <div
                                class="border border-white/5 rounded-lg p-6 text-center"
                            >
                                <p
                                    class="text-[10px] tracking-widest text-white/30 uppercase mb-3"
                                >
                                    Recognized Text
                                </p>
                                <p
                                    class="text-4xl font-black text-[#e8c84a] tracking-widest"
                                    style="font-family: 'Syne', sans-serif;"
                                >
                                    {recognizeResult.text ?? "—"}
                                </p>
                            </div>
                        {/if}
                    </div>
                {:else if activeTab === "video"}
                    <div class="space-y-6">
                        <div>
                            <p
                                class="text-[10px] tracking-[3px] text-white/30 uppercase mb-1"
                            >
                                Mode
                            </p>
                            <h2
                                class="text-xl font-bold"
                                style="font-family: 'Syne', sans-serif;"
                            >
                                Video Scan
                            </h2>
                            <p class="text-white/30 text-xs mt-1">
                                Upload a video and extract all unique plates
                                found across frames.
                            </p>
                        </div>

                        <label
                            class="block border border-dashed border-white/10 rounded-lg p-8 text-center cursor-pointer hover:border-[#e8c84a]/40 transition-colors group"
                        >
                            <input
                                type="file"
                                accept="video/*"
                                class="hidden"
                                onchange={(e) => handleFile(e, "video")}
                            />
                            <div
                                class="text-white/20 group-hover:text-white/40 transition-colors"
                            >
                                <div class="text-3xl mb-2">▶</div>
                                <div
                                    class="text-[11px] tracking-widest uppercase"
                                >
                                    {videoFile
                                        ? videoFile.name
                                        : "Drop video or click"}
                                </div>
                            </div>
                        </label>

                        <button
                            onclick={runVideo}
                            disabled={!videoFile || loading}
                            class="w-full py-3 bg-[#e8c84a] text-[#0a0a0a] text-[11px] tracking-[3px] uppercase font-bold rounded disabled:opacity-30 hover:bg-[#f0d060] transition-colors"
                        >
                            {loading ? "Scanning..." : "Scan Video"}
                        </button>

                        {#if videoResult}
                            <div
                                class="border border-white/5 rounded-lg overflow-hidden"
                            >
                                <div
                                    class="px-4 py-2 bg-white/3 border-b border-white/5 text-[10px] tracking-widest text-white/30 uppercase"
                                >
                                    Unique Plates Found
                                </div>
                                {#if Array.isArray(videoResult) && videoResult.length > 0}
                                    <div class="p-4 grid grid-cols-2 gap-2">
                                        {#each videoResult as det}
                                            <div
                                                class="flex items-center justify-between bg-white/3 rounded px-4 py-3"
                                            >
                                                <span
                                                    class="text-[#e8c84a] font-bold tracking-widest"
                                                    style="font-family: 'Syne', sans-serif;"
                                                    >{det.text}</span
                                                >
                                                <span
                                                    class="text-[10px] text-white/30"
                                                    >{(det.conf * 100).toFixed(
                                                        0,
                                                    )}%</span
                                                >
                                            </div>
                                        {/each}
                                    </div>
                                {:else}
                                    <p class="p-4 text-white/30 text-xs">
                                        No plates found.
                                    </p>
                                {/if}
                            </div>
                        {/if}
                    </div>
                {:else if activeTab === "webcam"}
                    <div class="space-y-6">
                        <div>
                            <p
                                class="text-[10px] tracking-[3px] text-white/30 uppercase mb-1"
                            >
                                Mode
                            </p>
                            <h2
                                class="text-xl font-bold"
                                style="font-family: 'Syne', sans-serif;"
                            >
                                Live Webcam
                            </h2>
                            <p class="text-white/30 text-xs mt-1">
                                Stream your camera and detect plates in real
                                time.
                            </p>
                        </div>

                        <div
                            class="border border-white/5 rounded-lg overflow-hidden bg-black aspect-video flex items-center justify-center"
                        >
                            <video
                                bind:this={videoEl}
                                autoplay
                                playsinline
                                class="w-full h-full object-contain {webcamActive
                                    ? ''
                                    : 'hidden'}"
                            ></video>
                            {#if !webcamActive}
                                <div
                                    class="text-white/20 text-[11px] tracking-widest uppercase"
                                >
                                    Camera off
                                </div>
                            {/if}
                        </div>

                        <button
                            onclick={webcamActive ? stopWebcam : startWebcam}
                            class="w-full py-3 text-[11px] tracking-[3px] uppercase font-bold rounded transition-colors {webcamActive
                                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/20'
                                : 'bg-[#e8c84a] text-[#0a0a0a] hover:bg-[#f0d060]'}"
                        >
                            {webcamActive ? "Stop Camera" : "Start Camera"}
                        </button>

                        {#if webcamResults.length > 0}
                            <div
                                class="border border-white/5 rounded-lg overflow-hidden"
                            >
                                <div
                                    class="px-4 py-2 bg-white/3 border-b border-white/5 text-[10px] tracking-widest text-white/30 uppercase"
                                >
                                    Plates Spotted
                                </div>
                                <div class="p-4 grid grid-cols-2 gap-2">
                                    {#each webcamResults as det}
                                        <div
                                            class="flex items-center justify-between bg-white/3 rounded px-4 py-3"
                                        >
                                            <span
                                                class="text-[#e8c84a] font-bold tracking-widest"
                                                style="font-family: 'Syne', sans-serif;"
                                                >{det.text}</span
                                            >
                                            <span
                                                class="text-[10px] text-white/30"
                                                >{(det.conf * 100).toFixed(
                                                    0,
                                                )}%</span
                                            >
                                        </div>
                                    {/each}
                                </div>
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>
        {/key}
    </main>
</div>
