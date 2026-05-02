# M6 hardware monitor — samples RAM, GPU usage, GPU VRAM, and disk
# C: activity every 2 seconds and writes one CSV-like line per sample
# to hw_log.txt. Run in a separate PowerShell window while the
# Atenia smoke test runs in another.
#
# Usage:
#     .\monitor_hw.ps1
# Stop with Ctrl+C. The output file is appended; delete it before a
# fresh run if you want a clean log:
#     Remove-Item hw_log.txt -ErrorAction SilentlyContinue
#
# Columns (tab-separated):
#   timestamp                  | ISO 8601 with milliseconds
#   ram_free_mb                | Win32_OperatingSystem.FreePhysicalMemory / 1024
#   ram_used_mb                | (TotalVisible - FreePhysical) / 1024
#   gpu_util_pct               | nvidia-smi utilization.gpu (RTX 4070 only)
#   gpu_vram_used_mb           | nvidia-smi memory.used
#   gpu_vram_total_mb          | nvidia-smi memory.total
#   disk_c_active_time_pct     | Win32_PerfFormattedData_PerfDisk_PhysicalDisk
#                                (LogicalDisk view: % active time on C:)

$ErrorActionPreference = 'Continue'
$logPath = Join-Path $PSScriptRoot 'hw_log.txt'

# Header (only if file is new or empty).
if (-not (Test-Path $logPath) -or (Get-Item $logPath).Length -eq 0) {
    "timestamp`tram_free_mb`tram_used_mb`tgpu_util_pct`tgpu_vram_used_mb`tgpu_vram_total_mb`tdisk_c_active_pct" |
        Out-File -FilePath $logPath -Encoding utf8
}

Write-Host "Monitoring hardware. Output: $logPath"
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

while ($true) {
    $timestamp = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fff')

    # RAM via WMI / CIM. FreePhysicalMemory and TotalVisibleMemorySize
    # are both in KiB; /1024 yields MiB.
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $ramFreeMb = [math]::Round($os.FreePhysicalMemory / 1024, 0)
        $ramUsedMb = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024, 0)
    } catch {
        $ramFreeMb = -1; $ramUsedMb = -1
    }

    # GPU via nvidia-smi. We query only the RTX 4070 (-i 0). If the
    # iGPU is enumerated as device 0 on this host the operator will
    # need to adjust -i; the default Windows ordering on this dev box
    # has the dGPU as 0 because the Intel iGPU is enumerated separately
    # under "Intel(R) UHD Graphics" in Task Manager.
    $gpuUtil = -1
    $gpuVramUsed = -1
    $gpuVramTotal = -1
    try {
        $smi = & nvidia-smi `
            --query-gpu=utilization.gpu,memory.used,memory.total `
            --format=csv,noheader,nounits `
            -i 0 2>$null
        if ($LASTEXITCODE -eq 0 -and $smi) {
            $parts = ($smi -split ',') | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 3) {
                $gpuUtil = [int]$parts[0]
                $gpuVramUsed = [int]$parts[1]
                $gpuVramTotal = [int]$parts[2]
            }
        }
    } catch {
        # nvidia-smi not on PATH or driver unavailable; columns stay -1.
    }

    # Disk C: % active time. Get-Counter is the most reliable Windows
    # API for this and works without admin rights for performance
    # counters.
    $diskActivePct = -1
    try {
        $samp = Get-Counter '\PhysicalDisk(0 C:)\% Disk Time' `
            -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
        $diskActivePct = [math]::Round($samp.CounterSamples[0].CookedValue, 1)
    } catch {
        # PhysicalDisk(0 C:) name varies by locale and disk count.
        # Try LogicalDisk(C:) as a fallback.
        try {
            $samp = Get-Counter '\LogicalDisk(C:)\% Disk Time' `
                -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
            $diskActivePct = [math]::Round($samp.CounterSamples[0].CookedValue, 1)
        } catch {
            $diskActivePct = -1
        }
    }

    $line = "{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}" -f `
        $timestamp, $ramFreeMb, $ramUsedMb, `
        $gpuUtil, $gpuVramUsed, $gpuVramTotal, `
        $diskActivePct

    Add-Content -Path $logPath -Value $line -Encoding utf8

    # Echo to console so the operator can see live values without
    # having to tail the file.
    Write-Host $line

    # Approximate 2 s cadence. `Get-Counter -SampleInterval 1` already
    # consumed ~1 s, so we add 1 s here.
    Start-Sleep -Seconds 1
}
