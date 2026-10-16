# Verify (md5) and extract the TUM-VI outdoors 512_16 sequences into H:\Research\datasets\tumvi
$ErrorActionPreference = 'Continue'
$arch = 'H:\Research\datasets-archive'
$tumvi = 'H:\Research\datasets\tumvi'
$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($i in 1..4) {
  $name = "dataset-outdoors${i}_512_16"
  $tar = Join-Path $arch "$name.tar"
  $expected = ((Get-Content (Join-Path $arch "$name.tar.md5") -Raw).Trim() -split '\s+')[0].ToLower()
  $actual = (Get-FileHash -Algorithm MD5 $tar).Hash.ToLower()
  if ($expected -ne $actual) { "$name MD5 MISMATCH expected $expected got $actual"; continue }
  "$name md5 ok t={0:n0}s" -f $sw.Elapsed.TotalSeconds
  if (Test-Path (Join-Path $tumvi "$name\mav0\cam0\data.csv")) { "$name already extracted"; continue }
  tar -xf $tar -C $tumvi
  "$name extracted t={0:n0}s" -f $sw.Elapsed.TotalSeconds
}
"ALL DONE t={0:n0}s" -f $sw.Elapsed.TotalSeconds
