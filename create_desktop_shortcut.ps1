$WshShell = New-Object -ComObject WScript.Shell
$desktopPaths = @(
    [Environment]::GetFolderPath('Desktop'),
    "C:\Users\hp\Desktop"
) | Select-Object -Unique

$projectDir = "C:\Users\hp\OneDrive\Desktop\CrowdPulse"
$targetBatch = Join-Path $projectDir "run_crowdpulse.bat"
$iconFile = Join-Path $projectDir "crowdpulse.ico"

foreach ($desk in $desktopPaths) {
    if (Test-Path $desk) {
        $shortcutPath = Join-Path $desk "CrowdPulse.lnk"
        $shortcut = $WshShell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $targetBatch
        $shortcut.WorkingDirectory = $projectDir
        $shortcut.IconLocation = "$iconFile,0"
        $shortcut.Description = "CrowdPulse - AI Crowd Density & Incident Management"
        $shortcut.Save()
        Write-Host "Created shortcut successfully at: $shortcutPath"
    }
}
