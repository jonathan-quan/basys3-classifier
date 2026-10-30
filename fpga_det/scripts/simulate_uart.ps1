$ErrorActionPreference='Stop'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
# Vivado's loader.bat silently exits 1 when PROCESSOR_ARCHITECTURE is missing from the environment.
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE='AMD64' }
$toolBin='C:\AMDDesignTools\2026.1\Vivado\bin'
# Golden packets come from the host client's own encoder plus the integer reference; the cycle count
# is the deterministic value measured by the core simulation.
$cycles=(Select-String -Path fpga_det/build/simulation_core.log -Pattern 'PASS FINAL case=0 .* cycles=(\d+)').Matches[0].Groups[1].Value
if (-not $cycles) { throw 'Run simulate_core.ps1 first (cycle count missing)' }
& .\.venv\Scripts\python.exe -B -m fpga_det.client --write-uart-vectors $cycles
if ($LASTEXITCODE -ne 0) { throw 'UART vector generation failed' }
& "$toolBin\xvlog.bat" --sv -i fpga_det/export fpga_det/rtl/detector_core.sv fpga/rtl/uart.sv fpga_det/rtl/basys3_detector.sv fpga_det/sim/tb_det_uart.sv
if ($LASTEXITCODE -ne 0) { throw 'RTL compilation failed' }
& "$toolBin\xelab.bat" tb_det_uart -s detector_uart_sim -O3
if ($LASTEXITCODE -ne 0) { throw 'RTL elaboration failed' }
$simulationOutput = & "$toolBin\xsim.bat" detector_uart_sim -runall 2>&1
$simulationOutput | Write-Output
if ($LASTEXITCODE -ne 0) { throw 'UART integration simulation failed' }
if (($simulationOutput -join "`n") -notmatch 'ALL UART FRAMING, CRC, RGB TO CHW, DETECTION AND OUTPUT BYTES MATCH') { throw 'Simulation did not reach its success assertions' }
$identity=Get-Content -LiteralPath fpga_det/export/hardware_identity.json -Raw | ConvertFrom-Json
@{model_id_hex=$identity.model_id_hex; passed=$true; cycles=[int]$cycles; checks='HELLO, bad CRC, bad image config, full 128x128 image -> exact 24-byte result packet'} | ConvertTo-Json | Set-Content -LiteralPath fpga_det/build/uart_simulation_pass.json
