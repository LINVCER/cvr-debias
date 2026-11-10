# 简化版调度：在同一 PowerShell 进程里用 Start-Job 实现并行
# 解释器默认用 PATH 中的 python；需要指定时设环境变量 PYTHON
$root = Split-Path -Parent $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { (Get-Command python).Source }
Set-Location $root
$log = Join-Path $root "artifacts\baselines_formal.log"
"" | Out-File $log -Encoding utf8

$models = @(
    @{model="mlp_cvr"; loss="bce"},
    @{model="esmm";   loss="esmm"},
    @{model="mmoe";   loss="esmm"},
    @{model="uc_escm"; loss="ips"},
    @{model="uc_escm"; loss="dr"},
    @{model="uc_escm"; loss="vrdr"}
)

$tStart = Get-Date
$total = 30

function Invoke-Run {
    param([string]$model, [string]$loss, [int]$seed, [string]$py, [string]$log)
    $cmd = "& '$py' -u -m src.train data=aliccp model=$model loss=$loss exp=main train.seed=$seed train.batch_size=16384 train.num_workers=2 train.device=cuda 2>&1 | Tee-Object -FilePath '$log' -Append"
    Invoke-Expression $cmd
}

$idx = 0
$jobs = @()
$maxParallel = 2

foreach ($m in $models) {
    foreach ($seed in 0..4) {
        # 跳过已完成
        $runDir = Join-Path $root "artifacts\runs\main\$($m.model)\seed$seed"
        if (Test-Path (Join-Path $runDir "metrics.json")) {
            "[$(Get-Date -Format HH:mm:ss)] SKIP $($m.model)+$($m.loss) seed$seed" | Out-File $log -Append; $idx++; continue
        }
        # 控制并发
        while ($jobs.Count -ge $maxParallel) {
            Start-Sleep -Seconds 15
            $done = @($jobs | Where-Object { $_.State -ne 'Running' })
            foreach ($j in $done) {
                Remove-Job $j -Force
            }
            $jobs = @($jobs | Where-Object { $_.State -eq 'Running' })
            $elapsed = (Get-Date) - $tStart
            "[$(Get-Date -Format HH:mm:ss)] PROGRESS  done=$idx/$total  running=$($jobs.Count)  elapsed=$([math]::Round($elapsed.TotalMinutes,1))min" | Out-File $log -Append
        }
        $jobName = "baseline_${($m.model)}_${($m.loss)}_$seed"
        $job = Start-Job -ScriptBlock {
            param($model, $loss, $seed, $py, $log, $root)
            Set-Location $root
            & $py -u -m src.train data=aliccp model=$model loss=$loss exp=main train.seed=$seed train.batch_size=16384 train.num_workers=2 train.device=cuda 2>&1 | Out-File $log -Append
        } -ArgumentList $m.model, $m.loss, $seed, $py, $log, $root -Name $jobName
        "[$(Get-Date -Format HH:mm:ss)] START  $($m.model)+$($m.loss) seed$seed  job=$jobName" | Out-File $log -Append
        $jobs += $job
        $idx++
    }
}

# 等所有 job 结束
while ($jobs.Count -gt 0) {
    Start-Sleep -Seconds 20
    $done = @($jobs | Where-Object { $_.State -ne 'Running' })
    foreach ($j in $done) {
        Receive-Job $j | Out-File $log -Append
        Remove-Job $j -Force
        "[$(Get-Date -Format HH:mm:ss)] DONE   $($j.Name)" | Out-File $log -Append
    }
    $jobs = @($jobs | Where-Object { $_.State -eq 'Running' })
}
"[$(Get-Date -Format HH:mm:ss)] === 全部完成 总耗时 $([math]::Round(((Get-Date) - $tStart).TotalMinutes,1)) min ===" | Out-File $log -Append
