$ErrorActionPreference = 'Continue'
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object Name,DriverVersion,AdapterRAM,VideoModeDescription
$disk = Get-PSDrive C | Select-Object Name,Used,Free
$record = [ordered]@{
  timestamp = (Get-Date).ToUniversalTime().ToString('o')
  computer = $env:COMPUTERNAME
  os = $os | Select-Object Caption,Version,BuildNumber,OSArchitecture,LastBootUpTime
  computer_system = $cs | Select-Object Manufacturer,Model,NumberOfLogicalProcessors,TotalPhysicalMemory
  gpu = @($gpu)
  nvidia_smi = (& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,pstate,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>&1) -join "`n"
  wsl_status = (& wsl --status 2>&1) -join "`n"
  wsl_list = (& wsl --list --verbose 2>&1) -join "`n"
  windows_python = (& python --version 2>&1) -join "`n"
  c_drive = $disk
}
$out = Join-Path $PSScriptRoot '..\results\windows_environment.json'
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $out
Get-Content $out
