$ErrorActionPreference='Stop'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
# Vivado's loader.bat silently exits 1 when PROCESSOR_ARCHITECTURE is missing from the environment.
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE='AMD64' }
$toolBin='C:\AMDDesignTools\2026.1\Vivado\bin'
& "$toolBin\xvlog.bat" --sv -i fpga_det/export fpga_det/rtl/detector_core.sv fpga_det/sim/tb_det_core.sv
if ($LASTEXITCODE -ne 0) { throw 'RTL compilation failed' }
& "$toolBin\xelab.bat" tb_det_core -s detector_core_sim -O3
if ($LASTEXITCODE -ne 0) { throw 'RTL elaboration failed' }
$simulationOutput = & "$toolBin\xsim.bat" detector_core_sim -runall 2>&1
$simulationOutput | Write-Output
New-Item -ItemType Directory -Path fpga_det/build -Force | Out-Null
$simulationOutput | Set-Content -LiteralPath fpga_det/build/simulation_core.log
if ($LASTEXITCODE -ne 0) { throw 'Integer-reference simulation failed' }
if (($simulationOutput -join "`n") -notmatch 'ALL DETECTOR LAYERS, POOL, HEAT, ATTENTION AND BOX OUTPUTS MATCH') { throw 'Simulation did not reach its success assertions' }
New-Item -ItemType Directory -Path fpga_det/build -Force | Out-Null
$identity=Get-Content -LiteralPath fpga_det/export/hardware_identity.json -Raw | ConvertFrom-Json
@{model_id_hex=$identity.model_id_hex; passed=$true; vectors=5; checks='Every backbone byte (11 layers), pooled, heat map, attention, class/presence/box outputs'} | ConvertTo-Json | Set-Content -LiteralPath fpga_det/build/simulation_pass.json
