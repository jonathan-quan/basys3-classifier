$ErrorActionPreference='Stop'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$toolBin='C:\AMDDesignTools\2026.1\Vivado\bin'
& "$toolBin\xvlog.bat" --sv -i fpga/export fpga_viz/rtl/classifier_core.sv fpga_viz/rtl/inference_telemetry.sv fpga_viz/rtl/uart.sv fpga_viz/rtl/basys3_classifier_viz.sv fpga_viz/sim/tb_viz_uart.sv
if ($LASTEXITCODE -ne 0) { throw 'Viz RTL compilation failed' }
& "$toolBin\xelab.bat" tb_viz_uart -s viz_uart_sim -debug typical
if ($LASTEXITCODE -ne 0) { throw 'Viz RTL elaboration failed' }
$simulationOutput = & "$toolBin\xsim.bat" viz_uart_sim -runall 2>&1
$simulationOutput | Write-Output
if ($LASTEXITCODE -ne 0 -or ($simulationOutput -join "`n") -notmatch 'ALL VIZ UART PACKETS MATCH') { throw 'Viz UART simulation failed' }
$identity=Get-Content -LiteralPath fpga/export/hardware_identity.json -Raw | ConvertFrom-Json
@{model_id_hex=$identity.model_id_hex;passed=$true;checks='original packets byte-identical; telemetry readout equals integer reference'} | ConvertTo-Json | Set-Content -LiteralPath fpga_viz/build/uart_simulation_pass.json
