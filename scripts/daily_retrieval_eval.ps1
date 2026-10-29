# Reruns the retrieval evaluation. It resumes from the disk cache (data/cache/retrieval.sqlite), so
# each run embeds as much as Gemini's free tier allows (about 1000 texts per day) and stops.
# Meant for the daily scheduled task "EuAiActCopilot-RetrievalEval"; when a run finishes
# completely (exit code 0, results/retrieval.json written) it removes that task.
# Output is appended to data/cache/daily_eval.log (git-ignored).

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root "data\cache\daily_eval.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

"=== $(Get-Date -Format o) ===" | Add-Content -Encoding utf8 $log
$env:PYTHONIOENCODING = "utf-8"
cmd /c "uv run --no-sync --env-file .env python evals/run_retrieval.py >> `"$log`" 2>&1"
$code = $LASTEXITCODE
"exit code: $code" | Add-Content -Encoding utf8 $log

if ($code -eq 0) {
    Unregister-ScheduledTask -TaskName "EuAiActCopilot-RetrievalEval" -Confirm:$false
    "complete: results written, scheduled task removed" | Add-Content -Encoding utf8 $log
}
