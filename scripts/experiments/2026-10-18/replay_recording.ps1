# Replay a phone recording through AR presets on the PC (Windows wrapper): settings for each preset from the recording's
# own settings.yaml (scripts/page_presets.mjs), then the side-by-side WASM replay in WSL (replay_recording.sh).
# Usage: replay_recording.ps1 <recording> [-Tag 05414e1] [-Seeds "1 2 3"] [-Presets fast,fastq,...]
param([Parameter(Mandatory)][string]$Rec, [string]$Tag = "05414e1", [string]$Seeds = "1 2 3",
      [string[]]$Presets = @("fast", "fastq", "fastqp", "gstart", "grecover", "small", "expo", "lean", "next"))
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$set = "H:\Research\results\replays\$Rec\settings"
New-Item -ItemType Directory -Force $set | Out-Null
node --experimental-default-type=module "$repo\scripts\page_presets.mjs" --base "H:\Research\datasets\phone\$Rec\settings.yaml" --name $Rec $set @Presets | Out-Null
wsl -d Ubuntu-22.04 -u root -- bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-18/replay_recording.sh $Tag $Rec "$Seeds" @Presets
