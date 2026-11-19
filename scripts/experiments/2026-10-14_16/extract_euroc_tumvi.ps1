# Extract EuRoC (ASL inner zips only, no ROS bags) and TUM-VI 512_16 rooms into H:\Research\datasets
$ErrorActionPreference = 'Continue'
$arch = 'H:\Research\datasets-archive'
$euroc = 'H:\Research\datasets\euroc'
$tumvi = 'H:\Research\datasets\tumvi'
$tmp = 'H:\Research\datasets\_tmp_extract'
New-Item -ItemType Directory -Force $euroc, $tumvi, $tmp | Out-Null
$sw = [Diagnostics.Stopwatch]::StartNew()

$seqs = @{
  'machine_hall' = @('MH_01_easy','MH_02_easy','MH_03_medium','MH_04_difficult','MH_05_difficult');
  'vicon_room1'  = @('V1_01_easy','V1_02_medium','V1_03_difficult');
  'vicon_room2'  = @('V2_01_easy','V2_02_medium','V2_03_difficult')
}
foreach ($outer in $seqs.Keys) {
  foreach ($s in $seqs[$outer]) {
    $inner = "$outer/$s/$s.zip"
    tar -xf "$arch\$outer.zip" -C $tmp $inner
    $dest = Join-Path $euroc $s
    New-Item -ItemType Directory -Force $dest | Out-Null
    tar -xf (Join-Path $tmp "$outer\$s\$s.zip") -C $dest
    Remove-Item (Join-Path $tmp "$outer\$s\$s.zip") -Force
    "{0} done t={1:n0}s" -f $s, $sw.Elapsed.TotalSeconds
  }
}
foreach ($i in 1..6) {
  tar -xf "$arch\dataset-room${i}_512_16.tar" -C $tumvi
  "room$i done t={0:n0}s" -f $sw.Elapsed.TotalSeconds
}
Remove-Item $tmp -Recurse -Force
"ALL DONE t={0:n0}s" -f $sw.Elapsed.TotalSeconds
