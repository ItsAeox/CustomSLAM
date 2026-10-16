param([string]$Expr = "JSON.stringify({ar: typeof window.__ar, coi: crossOriginIsolated, err: document.getElementById('err').textContent, go: document.getElementById('go').disabled, start: !document.getElementById('start').hidden, hint: document.getElementById('hint').textContent})")
# Evaluate a JS expression in the phone's localhost AR tab (Chrome DevTools protocol over adb-forwarded port 9222).
$tabs = Invoke-RestMethod -Uri "http://127.0.0.1:9222/json/list" -TimeoutSec 10
$tab = $tabs | Where-Object { $_.url -like 'http://localhost:8010/*' } | Select-Object -First 1
if (-not $tab) { "no localhost tab"; exit 1 }
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$ws.ConnectAsync([Uri]$tab.webSocketDebuggerUrl, [Threading.CancellationToken]::None).Wait(5000) | Out-Null
$msg = @{ id = 1; method = "Runtime.evaluate"; params = @{ expression = $Expr; returnByValue = $true; awaitPromise = $true } } | ConvertTo-Json -Compress -Depth 5
$bytes = [Text.Encoding]::UTF8.GetBytes($msg)
$ws.SendAsync([ArraySegment[byte]]::new($bytes), [Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).Wait(5000) | Out-Null
$buf = New-Object byte[] 262144; $sb = New-Object Text.StringBuilder
do {
  $r = $ws.ReceiveAsync([ArraySegment[byte]]::new($buf), [Threading.CancellationToken]::None); $r.Wait(15000) | Out-Null
  $sb.Append([Text.Encoding]::UTF8.GetString($buf, 0, $r.Result.Count)) | Out-Null
} while (-not $r.Result.EndOfMessage)
$ws.Dispose()
$resp = $sb.ToString() | ConvertFrom-Json
if ($resp.result.result.value) { $resp.result.result.value } else { $sb.ToString().Substring(0, [Math]::Min(600, $sb.Length)) }
