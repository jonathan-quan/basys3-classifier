# Detector + telemetry: (1) the original core testbench on the tapped core (all 11 layers, pooled,
# heat, attention and box outputs bit-exact on 5 frames); (2) full UART + telemetry testbench.
$ErrorActionPreference='Stop'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE='AMD64' }
$toolBin='C:\AMDDesignTools\2026.1\Vivado\bin'
New-Item -ItemType Directory -Path fpga_viz/det_build -Force | Out-Null
& "$toolBin\xvlog.bat" --sv -i fpga_det/export fpga_viz/det_rtl/detector_core.sv fpga_det/sim/tb_det_core.sv
if ($LASTEXITCODE -ne 0) { throw 'Core compilation failed' }
& "$toolBin\xelab.bat" tb_det_core -s viz_det_core_sim -O3
if ($LASTEXITCODE -ne 0) { throw 'Core elaboration failed' }
$core = & "$toolBin\xsim.bat" viz_det_core_sim -runall 2>&1
$core | Set-Content -LiteralPath fpga_viz/det_build/simulation_core.log
if ($LASTEXITCODE -ne 0 -or ($core -join "`n") -notmatch 'ALL DETECTOR LAYERS, POOL, HEAT, ATTENTION AND BOX OUTPUTS MATCH') { throw 'Core simulation failed' }
$identity=Get-Content -LiteralPath fpga_det/export/hardware_identity.json -Raw | ConvertFrom-Json
@{model_id_hex=$identity.model_id_hex; passed=$true; checks='tapped detector core: 11 layers, pooled, heat, attention, outputs on 5 frames'} | ConvertTo-Json | Set-Content -LiteralPath fpga_viz/det_build/simulation_pass.json
& .\.venv\Scripts\python.exe -B -m fpga_viz.make_det_vectors
if ($LASTEXITCODE -ne 0) { throw 'Vector generation failed' }
& "$toolBin\xvlog.bat" --sv -i fpga_det/export fpga_viz/det_rtl/detector_core.sv fpga_viz/det_rtl/detector_telemetry.sv fpga/rtl/uart.sv fpga_viz/det_rtl/basys3_detector_viz.sv fpga_viz/sim/tb_det_viz_uart.sv
if ($LASTEXITCODE -ne 0) { throw 'UART compilation failed' }
& "$toolBin\xelab.bat" tb_det_viz_uart -s viz_det_uart_sim -O3
if ($LASTEXITCODE -ne 0) { throw 'UART elaboration failed' }
$uart = & "$toolBin\xsim.bat" viz_det_uart_sim -runall 2>&1
$uart | Set-Content -LiteralPath fpga_viz/det_build/simulation_uart.log
if ($LASTEXITCODE -ne 0 -or ($uart -join "`n") -notmatch 'ALL DETECTOR VIZ UART PACKETS MATCH') { throw 'UART simulation failed' }
@{model_id_hex=$identity.model_id_hex; passed=$true; checks='original packets byte-identical; telemetry equals integer reference'} | ConvertTo-Json | Set-Content -LiteralPath fpga_viz/det_build/uart_simulation_pass.json
Write-Output 'DETECTOR VIZ SIMULATIONS PASSED'
