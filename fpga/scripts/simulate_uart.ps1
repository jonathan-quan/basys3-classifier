$ErrorActionPreference='Stop'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$toolBin='C:\AMDDesignTools\2026.1\Vivado\bin'
& "$toolBin\xvlog.bat" --sv -i fpga/export fpga/rtl/classifier_core.sv fpga/rtl/uart.sv fpga/rtl/basys3_classifier.sv fpga/sim/tb_uart.sv
if ($LASTEXITCODE -ne 0) { throw 'UART RTL compilation failed' }
& "$toolBin\xelab.bat" tb_uart -s classifier_uart_sim -debug typical
if ($LASTEXITCODE -ne 0) { throw 'UART RTL elaboration failed' }
$simulationOutput = & "$toolBin\xsim.bat" classifier_uart_sim -runall 2>&1
$simulationOutput | Write-Output
if ($LASTEXITCODE -ne 0 -or ($simulationOutput -join "`n") -notmatch 'ALL UART FRAMING, CRC, RGB TO CHW, INFERENCE AND OUTPUT BYTES MATCH') { throw 'UART integration simulation failed' }
$identity=Get-Content -LiteralPath fpga/export/hardware_identity.json -Raw | ConvertFrom-Json
@{model_id_hex=$identity.model_id_hex;passed=$true;checks='HELLO, CRC rejection, bad configuration, full RGB image packet and exact output'} | ConvertTo-Json | Set-Content -LiteralPath fpga/build/uart_simulation_pass.json
