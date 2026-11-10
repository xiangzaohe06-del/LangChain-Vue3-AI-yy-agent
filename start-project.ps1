$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtime = Join-Path $projectRoot '.runtime'
New-Item -ItemType Directory -Force $runtime | Out-Null

function Test-LocalPort([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $Port); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

function Start-ProjectProcess($Name, $Executable, $Arguments, $Directory) {
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $Directory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime "$Name.log") -RedirectStandardError (Join-Path $runtime "$Name-error.log") -PassThru
    $process.Id | Set-Content (Join-Path $runtime "$Name.pid")
    Write-Host "$Name launched (PID $($process.Id)); checking readiness."
}

function Wait-ProjectService([string]$Name, [int]$Port, [string]$ReadyUrl = '') {
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if (Test-LocalPort $Port) {
            if (-not $ReadyUrl) { Write-Host "$Name ready."; return }
            try {
                $response = Invoke-WebRequest -Uri $ReadyUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) { Write-Host "$Name ready."; return }
            } catch {}
        }
        $pidFile = Join-Path $runtime "$Name.pid"
        if (Test-Path $pidFile) {
            $serviceId = [int](Get-Content $pidFile)
            if (-not (Get-Process -Id $serviceId -ErrorAction SilentlyContinue)) {
                throw "$Name exited. Check .runtime/$Name-error.log and .runtime/$Name.log."
            }
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become ready. Check .runtime/$Name-error.log and .runtime/$Name.log."
}

if (-not (Test-LocalPort 27017)) {
    $mongo = Get-ChildItem (Join-Path $runtime 'mongodb/server') -Filter mongod.exe -Recurse | Select-Object -First 1
    if (-not $mongo) { throw 'MongoDB executable is missing from .runtime/mongodb/server.' }
    $data = Join-Path $runtime 'mongodb/data'
    New-Item -ItemType Directory -Force $data | Out-Null
    Start-ProjectProcess 'mongodb' $mongo.FullName @('--dbpath', ('"' + $data + '"'), '--bind_ip', '127.0.0.1', '--port', '27017') $projectRoot
}
Wait-ProjectService 'mongodb' 27017

if (-not (Test-LocalPort 8001)) {
    $python = Join-Path $projectRoot 'dietitian/.venv-win/Scripts/python.exe'
    if (-not (Test-Path $python)) { throw 'Windows Python environment is missing.' }
    $envFile = Join-Path $projectRoot 'dietitian/.env'
    $settings = @{}
    foreach ($line in (Get-Content $envFile)) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
            $settings[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
        }
    }
    if ($settings['RAG_VECTOR_BACKEND'] -eq 'milvus') {
        $milvusUri = $settings['MILVUS_URI']
        if (-not $milvusUri) { $milvusUri = 'http://127.0.0.1:19530' }
        if ($milvusUri -match '^http://(127\.0\.0\.1|localhost):19530/?$' -and -not (Test-LocalPort 19530)) {
            if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker is required for local Milvus.' }
            & docker compose -f (Join-Path $projectRoot 'docker-compose.milvus.yml') up -d
            if ($LASTEXITCODE -ne 0) { throw 'Milvus startup failed. Start Docker Desktop and check docker-compose.milvus.yml.' }
            Wait-ProjectService 'milvus' 19530 'http://127.0.0.1:9091/healthz'
        }
    }
    $missing = @('DASHSCOPE_BASE_URL', 'DASHSCOPE_API_KEY') | Where-Object { -not $settings[$_] }
    if (-not $settings['TAVILY_API_KEY'] -and -not $settings['TVILY_API_KEY']) {
        $missing = @($missing) + 'TAVILY_API_KEY (or TVILY_API_KEY)'
    }
    if ($missing) {
        Write-Warning ('Fill dietitian/.env before starting the backend: ' + ($missing -join ', '))
    } else {
        Start-ProjectProcess 'backend' $python @('-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8001') (Join-Path $projectRoot 'dietitian')
    }
}

if (Test-Path (Join-Path $runtime 'backend.pid')) {
    if (-not $missing) {
        Wait-ProjectService 'backend' 8001 'http://127.0.0.1:8001/openapi.json'
    }
}

if (-not (Test-LocalPort 80)) {
    Start-ProjectProcess 'frontend' (Get-Command node.exe).Source @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '80', '--strictPort') (Join-Path $projectRoot 'dietitian-ui')
}
Wait-ProjectService 'frontend' 80 'http://127.0.0.1/'

Write-Host 'Frontend: http://127.0.0.1/'
Write-Host 'Backend docs (when running): http://127.0.0.1:8001/docs'
Write-Host 'Logs: .runtime/'
