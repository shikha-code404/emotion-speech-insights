# Starts the backend (port 8000) and frontend (port 5173) in two new windows.
# Usage, from the project folder:   .\start.ps1
# Close those two windows (or press Ctrl+C in them) to stop the servers.

$root = $PSScriptRoot

foreach ($port in 8000, 5173) {
    $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        $proc = Get-Process -Id ($busy | Select-Object -First 1).OwningProcess -ErrorAction SilentlyContinue
        Write-Host "Port $port is already in use by '$($proc.ProcessName)' (PID $($proc.Id))." -ForegroundColor Yellow
        Write-Host "Close that window first, or run:  Stop-Process -Id $($proc.Id)" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
    Write-Host "No .venv found. Create it first: python -m venv .venv; .\.venv\Scripts\pip install -r backend\requirements.txt" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "Installing frontend packages (first run only)..."
    Push-Location "$root\frontend"; npm install; Pop-Location
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; `$Host.UI.RawUI.WindowTitle = 'Backend :8000'; .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; `$Host.UI.RawUI.WindowTitle = 'Frontend :5173'; npm run dev"

Write-Host "Starting... open http://localhost:5173 in a few seconds." -ForegroundColor Green
