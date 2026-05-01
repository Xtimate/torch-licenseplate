<script lang="ts">
    const API_BASE = "http://localhost:8000";

    let activeTab = "pipeline";
    let pipelineFile = null;
    let detectFile = null;
    let recognizeFile = null;
    let videoFile = null;

    let pipelineResult = null;
    let detectResult = null;
    let recognizeResult = null;
    let videoResult = null;

    let loading = false;
    let previewUrl = null;

    function setTab(tab: string) {
        activeTab = tab;
        previewUrl = null;
    }
</script>
