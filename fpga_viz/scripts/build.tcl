# Build of the fpga_viz design (original classifier + observation-only telemetry).
# Same gates as fpga/scripts/build.tcl; also exports the routed placement for the chip map.
set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
set_param general.maxThreads 8
file mkdir fpga_viz/build
file delete -force fpga_viz/build/build_status.json fpga_viz/build/basys3_classifier_viz.bit
set rtl [list fpga_viz/rtl/classifier_core.sv fpga_viz/rtl/inference_telemetry.sv fpga_viz/rtl/uart.sv fpga_viz/rtl/basys3_classifier_viz.sv]
if {![file exists fpga_viz/build/simulation_pass.json]} {error "Viz core simulation has not passed"}
if {![file exists fpga_viz/build/uart_simulation_pass.json]} {error "Viz UART/telemetry simulation has not passed"}
foreach file [list fpga_viz/rtl/classifier_core.sv fpga/export/layers.svh fpga/sim/tb_core.sv] {
    if {[file mtime $file]>[file mtime fpga_viz/build/simulation_pass.json]} {error "RTL/export changed after core simulation: $file"}
}
foreach file [concat $rtl [list fpga_viz/sim/tb_viz_uart.sv]] {
    if {[file mtime $file]>[file mtime fpga_viz/build/uart_simulation_pass.json]} {error "RTL changed after UART simulation: $file"}
}
set fd [open fpga/export/hardware_identity.json r];set identity [read $fd];close $fd
regexp {"model_id_hex": "([0-9a-f]+)"} $identity all model_id
foreach marker {simulation_pass.json uart_simulation_pass.json} {
    set fd [open fpga_viz/build/$marker r];set simulation [read $fd];close $fd
    if {![string match "*$model_id*" $simulation]} {error "Simulation/export identity mismatch in $marker"}
}
create_project -in_memory -part xc7a35tcpg236-1
set_property include_dirs [list [file normalize fpga/export]] [current_fileset]
read_verilog -sv $rtl
read_xdc fpga_viz/constraints/basys3_viz.xdc
synth_design -top basys3_classifier_viz -part xc7a35tcpg236-1 -flatten_hierarchy rebuilt
report_utilization -file fpga_viz/build/utilization_synth.rpt
write_checkpoint -force fpga_viz/build/post_synth.dcp

set lutram [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAM32* || REF_NAME =~ RAM64* || REF_NAME =~ RAM128* || REF_NAME =~ RAM256* || REF_NAME =~ RAMD* || REF_NAME =~ RAMS*}]
puts "SYNTH_LUTRAM_PRIMITIVES [llength $lutram]"
foreach bank {activation_a activation_b} {
    set cells [get_cells -hierarchical -quiet -filter "REF_NAME =~ RAMB* && NAME =~ *$bank*"]
    puts "SYNTH_BRAM_PRIMITIVES_$bank [llength $cells]"
    if {[llength $cells]==0} {error "Activation bank $bank did not map to block RAM; refusing implementation"}
}
if {[llength $lutram]>1024} {error "Unexpected LUTRAM count [llength $lutram]"}

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force fpga_viz/build/post_route.dcp
report_utilization -file fpga_viz/build/utilization.rpt
report_utilization -hierarchical -file fpga_viz/build/utilization_hier.rpt
report_ram_utilization -file fpga_viz/build/ram_utilization.rpt
report_timing_summary -report_unconstrained -max_paths 10 -file fpga_viz/build/timing.rpt
report_route_status -file fpga_viz/build/route_status.rpt
report_drc -name final_drc -file fpga_viz/build/drc.rpt

set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1 -nworst 1]
if {[llength $setup_paths]==0 || [llength $hold_paths]==0} {error "No timed paths; refusing bitstream"}
set wns [get_property SLACK [lindex $setup_paths 0]]
set whs [get_property SLACK [lindex $hold_paths 0]]
puts "IMPLEMENTED_WNS_NS $wns"
puts "IMPLEMENTED_WHS_NS $whs"
puts "WORST_SETUP_PATH [get_property STARTPOINT_PIN [lindex $setup_paths 0]] -> [get_property ENDPOINT_PIN [lindex $setup_paths 0]]"
set unrouted [llength [get_nets -hierarchical -quiet -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == CONFLICTS}]]
set drc_errors [llength [get_drc_violations -quiet -name final_drc -filter {SEVERITY == Error}]]
puts "UNROUTED_OR_CONFLICT_NETS $unrouted DRC_ERRORS $drc_errors"
if {$wns<0} {error "100 MHz setup timing failed (WNS $wns ns)"}
if {$whs<0} {error "Hold timing failed (WHS $whs ns)"}
if {$unrouted>0} {error "Design is not fully routed"}
if {$drc_errors>0} {error "Routed design has DRC errors"}

source fpga_viz/scripts/placement_procs.tcl
export_placement [file normalize fpga_viz/build/placement]

set lut [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ LUT*}]]
set ff [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ FD*}]]
set dsp [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ DSP48*}]]
set ramb36 [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB36*}]]
set ramb18 [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB18*}]]
write_bitstream -force fpga_viz/build/basys3_classifier_viz.bit
set fd [open fpga_viz/build/build_status.json w]
puts $fd "{\"model_id_hex\":\"$model_id\",\"part\":\"xc7a35tcpg236-1\",\"clock_hz\":100000000,\"wns_ns\":$wns,\"whs_ns\":$whs,\"drc_errors\":$drc_errors,\"lut_cells\":$lut,\"ff_cells\":$ff,\"dsp_cells\":$dsp,\"ramb36\":$ramb36,\"ramb18\":$ramb18,\"bitstream\":\"fpga_viz/build/basys3_classifier_viz.bit\"}"
close $fd
puts "BITSTREAM_READY fpga_viz/build/basys3_classifier_viz.bit MODEL_ID $model_id"
exit
