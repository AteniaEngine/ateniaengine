param(
    [string]$ModelsRoot = "D:\models",
    [string]$WorkRoot = "D:\Atenia",
    [ValidateSet("quick", "full")]
    [string]$Suite = "quick",
    [int]$MaxTokens = 32,
    [string]$Prompt = "Tell me about the history of Rome",
    [switch]$SkipBuild,
    [switch]$ListOnly,
    [string]$ExePath = ""
)

$ErrorActionPreference = "Stop"

function New-Dir($Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Write-Utf8Json($Path, $Object) {
    $Object | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Parse-Plan($StderrText) {
    $plan = [ordered]@{
        vram_tensors = $null
        vram_gib = $null
        ram_tensors = $null
        ram_gib = $null
        disk_tensors = $null
        disk_gib = $null
        manifest_path = $null
        recommended_mode = $null
    }

    if ($StderrText -match 'Numeric contract:\s*(.+?model\.numcert\.json).*?recommended mode:\s*([A-Za-z0-9_]+)') {
        $plan.manifest_path = $Matches[1].Trim()
        $plan.recommended_mode = $Matches[2].Trim()
    }
    if ($StderrText -match 'VRAM:\s+(\d+)\s+tensors\s+\(([0-9.]+)\s+GiB\)') {
        $plan.vram_tensors = [int]$Matches[1]
        $plan.vram_gib = [double]$Matches[2]
    }
    if ($StderrText -match 'RAM:\s+(\d+)\s+tensors\s+\(([0-9.]+)\s+GiB\)') {
        $plan.ram_tensors = [int]$Matches[1]
        $plan.ram_gib = [double]$Matches[2]
    }
    if ($StderrText -match 'Disk:\s+(\d+)\s+tensors\s+\(([0-9.]+)\s+GiB\)') {
        $plan.disk_tensors = [int]$Matches[1]
        $plan.disk_gib = [double]$Matches[2]
    }
    return $plan
}

function Get-NvidiaSnapshot {
    $snapshot = [ordered]@{
        nvidia_smi = $null
        gpu_query = @()
        driver_query_error = $null
    }
    try {
        $snapshot.nvidia_smi = (& nvidia-smi 2>$null) -join "`n"
    } catch {
        $snapshot.driver_query_error = $_.Exception.Message
    }
    try {
        $rows = & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit --format=csv,noheader,nounits 2>$null
        foreach ($row in $rows) {
            $cols = $row -split '\s*,\s*'
            if ($cols.Count -ge 6) {
                $snapshot.gpu_query += [ordered]@{
                    name = $cols[0]
                    driver_version = $cols[1]
                    memory_total_mib = [int]$cols[2]
                    memory_free_mib = [int]$cols[3]
                    temperature_c = [int]$cols[4]
                    power_limit_w = [double]$cols[5]
                }
            }
        }
    } catch {
        $snapshot.driver_query_error = $_.Exception.Message
    }
    return $snapshot
}

function Get-SystemSnapshot {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    return [ordered]@{
        computer_name = $env:COMPUTERNAME
        timestamp = (Get-Date).ToString("o")
        os_caption = $os.Caption
        os_version = $os.Version
        cpu_name = $cpu.Name
        logical_processors = $cpu.NumberOfLogicalProcessors
        ram_total_gib = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        ram_free_gib = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        models_root = $ModelsRoot
        work_root = $WorkRoot
        suite = $Suite
        max_tokens = $MaxTokens
        prompt = $Prompt
    }
}

function New-Case($Name, $Kind, $RelPath, $Mode, $Tags = @()) {
    return [ordered]@{
        name = $Name
        kind = $Kind
        rel_path = $RelPath
        mode = $Mode
        tags = $Tags
    }
}

if (-not (Test-Path -LiteralPath $ModelsRoot)) {
    throw "ModelsRoot does not exist: $ModelsRoot"
}

$cases = @()

# GGUF quantized certification path (M11.D.5).
$cases += New-Case "gguf_tinyllama_q4_quantized" "gguf" "TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF" "manifest" @("m11d", "small")
$cases += New-Case "gguf_tinyllama_q8_quantized" "gguf" "tinyllama-q8_0" "manifest" @("m11d", "small")
$cases += New-Case "gguf_llama32_1b_q4_quantized" "gguf" "Llama-3.2-1B-Instruct-Q4_K_M-GGUF" "manifest" @("m11d", "1b")
$cases += New-Case "gguf_smollm2_q4_quantized" "gguf" "SmolLM2-1.7B-Instruct-GGUF" "manifest" @("m11d", "1b")
$cases += New-Case "gguf_phi35_q4_quantized" "gguf" "Phi-3.5-mini-instruct-Q4_K_M-GGUF" "manifest" @("m11d", "phi")

# Safetensors reference family. Run both manifest/default and fast override.
foreach ($model in @(
    @{slug="tinyllama"; rel="tinyllama-1.1b"; tags=@("m46", "small")},
    @{slug="smollm2"; rel="smollm2-1.7b-instruct"; tags=@("m46", "1b")},
    @{slug="qwen25_15b"; rel="qwen2.5-1.5b-instruct"; tags=@("m46", "1b")},
    @{slug="llama32_1b"; rel="llama-3.2-1b-instruct"; tags=@("m46", "1b")}
)) {
    $cases += New-Case ("st_" + $model.slug + "_manifest") "safetensors" $model.rel "manifest" $model.tags
    $cases += New-Case ("st_" + $model.slug + "_fast") "safetensors" $model.rel "fast" ($model.tags + @("fast"))
}

if ($Suite -eq "full") {
    foreach ($model in @(
        @{slug="phi35"; rel="phi-3.5-mini-instruct"; tags=@("top10", "phi")},
        @{slug="mistral7b"; rel="mistral-7b-v0.3"; tags=@("top10", "7b")},
        @{slug="gemma2_2b"; rel="gemma-2-2b-it"; tags=@("top10", "2b")},
        @{slug="falcon3_7b"; rel="falcon3-7b-instruct"; tags=@("top10", "7b")},
        @{slug="llama2_13b"; rel="llama-2-13b-chat"; tags=@("13b", "beyond-vram")}
    )) {
        $cases += New-Case ("st_" + $model.slug + "_manifest") "safetensors" $model.rel "manifest" $model.tags
        $cases += New-Case ("st_" + $model.slug + "_fast") "safetensors" $model.rel "fast" ($model.tags + @("fast"))
    }
}

if ($ListOnly) {
    $cases | ForEach-Object {
        $modelPath = Join-Path $ModelsRoot $_.rel_path
        [pscustomobject][ordered]@{
            name = $_.name
            kind = $_.kind
            mode = $_.mode
            model_path = $modelPath
            exists = Test-Path -LiteralPath $modelPath
            tags = ($_.tags -join ",")
        }
    } | Format-Table -AutoSize
    return
}

$targetDir = Join-Path $WorkRoot "cargo-target-rtx3090"
$cacheRoot = Join-Path $WorkRoot "runtime-cache"
$runId = "rtx3090_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$outRoot = Join-Path (Join-Path $WorkRoot "bench_logs") $runId
$rawRoot = Join-Path $outRoot "raw"
New-Dir $targetDir
New-Dir $cacheRoot
New-Dir $rawRoot

if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Join-Path $targetDir "release\atenia.exe"
}

if (-not $SkipBuild) {
    cargo build --target-dir $targetDir --release --bin atenia
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "atenia.exe not found: $ExePath"
}

$runMeta = [ordered]@{
    run_id = $runId
    command_line = $MyInvocation.Line
    script = $PSCommandPath
    repo_root = (Get-Location).Path
    exe_path = $ExePath
    target_dir = $targetDir
    cache_root = $cacheRoot
    output_root = $outRoot
    system = Get-SystemSnapshot
    nvidia = Get-NvidiaSnapshot
    cases_requested = $cases.Count
}
Write-Utf8Json (Join-Path $outRoot "run_metadata.json") $runMeta

$summary = @()
$jsonlPath = Join-Path $outRoot "summary.jsonl"
if (Test-Path $jsonlPath) { Remove-Item -LiteralPath $jsonlPath -Force }

foreach ($case in $cases) {
    $modelPath = Join-Path $ModelsRoot $case.rel_path
    $stdoutPath = Join-Path $rawRoot ($case.name + ".stdout.json")
    $stderrPath = Join-Path $rawRoot ($case.name + ".stderr.log")
    $envPath = Join-Path $rawRoot ($case.name + ".env.json")
    $caseCache = Join-Path $cacheRoot $case.name
    New-Dir $caseCache

    $beforeGpu = Get-NvidiaSnapshot
    $caseEnv = [ordered]@{
        ATENIA_MODELS_ROOT = $ModelsRoot
        ATENIA_DISK_TIER_DIR = $caseCache
        ATENIA_M8_BF16_KERNEL = "1"
        ATENIA_FAST_MODE = $null
    }

    $oldModelsRoot = $env:ATENIA_MODELS_ROOT
    $oldDiskTier = $env:ATENIA_DISK_TIER_DIR
    $oldBf16 = $env:ATENIA_M8_BF16_KERNEL
    $oldFast = $env:ATENIA_FAST_MODE

    $env:ATENIA_MODELS_ROOT = $ModelsRoot
    $env:ATENIA_DISK_TIER_DIR = $caseCache
    $env:ATENIA_M8_BF16_KERNEL = "1"
    if ($case.mode -eq "fast") {
        $env:ATENIA_FAST_MODE = "1"
        $caseEnv.ATENIA_FAST_MODE = "1"
    } else {
        Remove-Item Env:ATENIA_FAST_MODE -ErrorAction SilentlyContinue
    }
    Write-Utf8Json $envPath $caseEnv

    $result = [ordered]@{
        run_id = $runId
        name = $case.name
        kind = $case.kind
        mode = $case.mode
        tags = $case.tags
        model_path = $modelPath
        exists = Test-Path -LiteralPath $modelPath
        skipped = $false
        exit_code = $null
        json_parse_ok = $false
        tokens_generated = $null
        total_seconds = $null
        tokens_per_second = $null
        eos_reached = $null
        generated_text_prefix = $null
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        env_path = $envPath
        cache_dir = $caseCache
        plan = $null
        before_gpu = $beforeGpu.gpu_query
        after_gpu = $null
        error = $null
    }

    if (-not $result.exists) {
        $result.skipped = $true
        $result.error = "model directory missing"
    } else {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            & $ExePath generate --prompt $Prompt --model $modelPath --max-tokens $MaxTokens --output json --no-progress 2> $stderrPath > $stdoutPath
            $result.exit_code = $LASTEXITCODE
        } catch {
            $result.exit_code = $LASTEXITCODE
            $result.error = $_.Exception.Message
        }
        $sw.Stop()

        $stderrText = ""
        if (Test-Path -LiteralPath $stderrPath) {
            $stderrText = Get-Content -LiteralPath $stderrPath -Raw
        }
        $result.plan = Parse-Plan $stderrText
        $result.wall_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        $result.after_gpu = (Get-NvidiaSnapshot).gpu_query

        try {
            $json = Get-Content -LiteralPath $stdoutPath -Raw | ConvertFrom-Json
            $result.json_parse_ok = $true
            $result.tokens_generated = [int]$json.tokens_generated
            $result.total_seconds = [double]$json.total_seconds
            $result.tokens_per_second = [double]$json.tokens_per_second
            $result.eos_reached = [bool]$json.eos_reached
            $text = [string]$json.generated_text
            $result.generated_text_prefix = if ($text.Length -gt 160) { $text.Substring(0, 160) } else { $text }
        } catch {
            $result.json_parse_ok = $false
            if (-not $result.error) {
                $result.error = "json parse failed: " + $_.Exception.Message
            }
        }
    }

    if ($null -ne $oldModelsRoot) { $env:ATENIA_MODELS_ROOT = $oldModelsRoot } else { Remove-Item Env:ATENIA_MODELS_ROOT -ErrorAction SilentlyContinue }
    if ($null -ne $oldDiskTier) { $env:ATENIA_DISK_TIER_DIR = $oldDiskTier } else { Remove-Item Env:ATENIA_DISK_TIER_DIR -ErrorAction SilentlyContinue }
    if ($null -ne $oldBf16) { $env:ATENIA_M8_BF16_KERNEL = $oldBf16 } else { Remove-Item Env:ATENIA_M8_BF16_KERNEL -ErrorAction SilentlyContinue }
    if ($null -ne $oldFast) { $env:ATENIA_FAST_MODE = $oldFast } else { Remove-Item Env:ATENIA_FAST_MODE -ErrorAction SilentlyContinue }

    $summary += [pscustomobject]$result
    ($result | ConvertTo-Json -Depth 12 -Compress) | Add-Content -LiteralPath $jsonlPath -Encoding UTF8

    $status = if ($result.skipped) { "SKIP" } elseif ($result.exit_code -eq 0 -and $result.json_parse_ok) { "OK" } else { "FAIL" }
    $tps = if ($null -ne $result.tokens_per_second) { "{0:N3}" -f $result.tokens_per_second } else { "-" }
    Write-Host ("[{0}] {1} exit={2} json={3} tok={4} tps={5}" -f $status, $case.name, $result.exit_code, $result.json_parse_ok, $result.tokens_generated, $tps)
}

$csvPath = Join-Path $outRoot "summary.csv"
$summaryRows = foreach ($r in $summary) {
    [pscustomobject][ordered]@{
        name = $r.name
        kind = $r.kind
        mode = $r.mode
        exists = $r.exists
        skipped = $r.skipped
        exit_code = $r.exit_code
        json_parse_ok = $r.json_parse_ok
        tokens_generated = $r.tokens_generated
        total_seconds = $r.total_seconds
        tokens_per_second = $r.tokens_per_second
        eos_reached = $r.eos_reached
        vram_tensors = $r.plan.vram_tensors
        vram_gib = $r.plan.vram_gib
        ram_tensors = $r.plan.ram_tensors
        ram_gib = $r.plan.ram_gib
        disk_tensors = $r.plan.disk_tensors
        disk_gib = $r.plan.disk_gib
        recommended_mode = $r.plan.recommended_mode
        model_path = $r.model_path
        error = $r.error
    }
}
$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $csvPath

$mdPath = Join-Path $outRoot "summary.md"
$ok = @($summary | Where-Object { -not $_.skipped -and $_.exit_code -eq 0 -and $_.json_parse_ok }).Count
$fail = @($summary | Where-Object { -not $_.skipped -and -not ($_.exit_code -eq 0 -and $_.json_parse_ok) }).Count
$skip = @($summary | Where-Object { $_.skipped }).Count

$md = @()
$md += "# Atenia RTX 3090 battery"
$md += ""
$md += "- Run id: ``$runId``"
$md += "- Suite: ``$Suite``"
$md += "- Models root: ``$ModelsRoot``"
$md += "- Work root: ``$WorkRoot``"
$md += "- Prompt: ``$Prompt``"
$md += "- Max tokens: ``$MaxTokens``"
$md += "- OK / fail / skip: **$ok / $fail / $skip**"
$md += ""
$md += "## Results"
$md += ""
$md += "| Case | Kind | Mode | Exit | JSON | Tokens | tok/s | VRAM | RAM | Disk | Text prefix |"
$md += "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |"
foreach ($r in $summary) {
    $jsonOk = if ($r.json_parse_ok) { "yes" } else { "no" }
    $tokens = if ($null -ne $r.tokens_generated) { $r.tokens_generated } else { "" }
    $tps = if ($null -ne $r.tokens_per_second) { "{0:N3}" -f $r.tokens_per_second } else { "" }
    $vram = if ($null -ne $r.plan.vram_tensors) { "$($r.plan.vram_tensors) / $($r.plan.vram_gib) GiB" } else { "" }
    $ram = if ($null -ne $r.plan.ram_tensors) { "$($r.plan.ram_tensors) / $($r.plan.ram_gib) GiB" } else { "" }
    $disk = if ($null -ne $r.plan.disk_tensors) { "$($r.plan.disk_tensors) / $($r.plan.disk_gib) GiB" } else { "" }
    $prefix = ([string]$r.generated_text_prefix).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
    $md += "| `$($r.name)` | $($r.kind) | $($r.mode) | $($r.exit_code) | $jsonOk | $tokens | $tps | $vram | $ram | $disk | $prefix |"
}
$md += ""
$md += "## Files"
$md += ""
$md += "- Metadata: ``$($runMeta.output_root)\run_metadata.json``"
$md += "- JSONL: ``$jsonlPath``"
$md += "- CSV: ``$csvPath``"
$md += "- Raw stdout/stderr: ``$rawRoot``"
$md -join "`n" | Set-Content -LiteralPath $mdPath -Encoding UTF8

Write-Host ""
Write-Host "Battery complete."
Write-Host "Summary: $mdPath"
Write-Host "CSV:     $csvPath"
Write-Host "JSONL:   $jsonlPath"
