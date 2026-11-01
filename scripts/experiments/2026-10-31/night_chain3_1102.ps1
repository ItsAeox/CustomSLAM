# 2 Nov early chain 3 (PowerShell: the renderer runs on Windows Python): after night_chain2_1102.sh, render S13
# flatfirst0-5 (6 at a time, ~30 min), then flatfirst_run.sh <tag> in WSL. Log: results/night_chain3_1102.log
# Usage: powershell -File night_chain3_1102.ps1 1f4b239
param([string]$Tag)
$L = 'H:\Research\results\night_chain2_1102.log'
$O = 'H:\Research\results\night_chain3_1102.log'
while (-not (Select-String -Path $L -Pattern '=== chain 2 done|FAILED|not started|missing' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
"=== chain 2 ended $(Get-Date -Format T): $((Get-Content $L)[-1])" | Out-File -Append -Encoding utf8 $O
if (Select-String -Path $L -Pattern 'FAILED|not started|missing' -Quiet) { "=== chain 2 did not build $($Tag), flatfirst not started" | Out-File -Append -Encoding utf8 $O; exit 1 }
Set-Location H:\Research\CustomSLAM
"=== render flatfirst $(Get-Date -Format T)" | Out-File -Append -Encoding utf8 $O
python synth/make_dataset.py --only flatfirst0,flatfirst1,flatfirst2,flatfirst3,flatfirst4,flatfirst5 --procs 6 *>> H:\Research\results\render_flatfirst.log
"=== render done $(Get-Date -Format T)" | Out-File -Append -Encoding utf8 $O
wsl -d Ubuntu-22.04 -u root -- bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-31/flatfirst_run.sh $Tag *>> $O
"=== chain 3 done $(Get-Date -Format T)" | Out-File -Append -Encoding utf8 $O
