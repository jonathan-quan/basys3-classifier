# The viz copy of the core must still reproduce every byte of all 9 layers, the pooled
# vector, logits and score on the 3 original reference images (unchanged tb_core.sv).
$ErrorActionPreference='Stop'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$toolBin='C:\AMDDesignTools\2026.1\Vivado\bin'
& "$toolBin\xvlog.bat" --sv -i fpga/export fpga_viz/rtl/classifier_core.sv fpga/sim/tb_core.sv
if ($LASTEXITCODE -ne 0) { throw 'RTL compilation failed' }
& "$toolBin\xelab.bat" tb_core -s viz_core_sim -debug typical
if ($LASTEXITCODE -ne 0) { throw 'RTL elaboration failed' }
$simulationOutput = & "$toolBin\xsim.bat" viz_core_sim -runall 2>&1
$simulationOutput | Write-Output
if ($LASTEXITCODE -ne 0) { throw 'Integer-reference simulation failed' }
if (($simulationOutput -join "`n") -notmatch 'ALL INTEGER LAYERS, POOL, LOGITS AND CONFIDENCE MATCH') { throw 'Simulation did not reach its success assertions' }
New-Item -ItemType Directory -Path fpga_viz/build -Force | Out-Null
$identity=Get-Content -LiteralPath fpga/export/hardware_identity.json -Raw | ConvertFrom-Json
@{model_id_hex=$identity.model_id_hex; passed=$true; vectors=3; checks='viz core: every convolution byte, pooled byte, logit and confidence'} | ConvertTo-Json | Set-Content -LiteralPath fpga_viz/build/simulation_pass.json
