$ErrorActionPreference = 'Stop'
$runtimePath = Join-Path $PSScriptRoot '.runtime'

function Test-MongoPort {
    $connection = New-Object System.Net.Sockets.TcpClient
    try { $connection.Connect('127.0.0.1', 27017); return $true }
    catch { return $false }
    finally { $connection.Dispose() }
}

if (Test-MongoPort) {
    Write-Host 'Port 27017 is already listening.'
    return
}

$mongoExecutable = Get-ChildItem -LiteralPath (Join-Path $runtimePath 'mongodb/server') -Filter mongod.exe -Recurse | Select-Object -First 1
if (-not $mongoExecutable) { throw 'MongoDB executable not found in .runtime/mongodb/server.' }

$dataPath = Join-Path $runtimePath 'mongodb/data'
New-Item -ItemType Directory -Force -Path $dataPath | Out-Null
$mongoProcess = Start-Process -FilePath $mongoExecutable.FullName -ArgumentList @('--dbpath', ('"' + $dataPath + '"'), '--bind_ip', '127.0.0.1', '--port', '27017') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'mongodb.log') -RedirectStandardError (Join-Path $runtimePath 'mongodb-error.log') -PassThru
$mongoProcess.Id | Set-Content -LiteralPath (Join-Path $runtimePath 'mongodb.pid')

for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if (Test-MongoPort) {
        Write-Host 'MongoDB ready at 127.0.0.1:27017.'
        return
    }
    if ($mongoProcess.HasExited) { throw 'MongoDB exited. Check .runtime/mongodb.log and mongodb-error.log.' }
    Start-Sleep -Milliseconds 500
}
throw 'MongoDB startup timed out. Check .runtime/mongodb.log and mongodb-error.log.'
