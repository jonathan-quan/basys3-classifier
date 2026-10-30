set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
set_param general.maxThreads 8
file mkdir fpga/build
# Remove stale success artifacts so a failed run can never leave an old "ready" state behind.
file delete -force fpga/build/build_status.json fpga/build/basys3_classifier.bit
if {![file exists fpga/build/simulation_pass.json]} {error "Integer-reference simulation has not passed"}
if {![file exists fpga/build/uart_simulation_pass.json]} {error "UART integration simulation has not passed"}
foreach file [list fpga/rtl/classifier_core.sv fpga/export/layers.svh fpga/sim/tb_core.sv] {
    if {[file mtime $file]>[file mtime fpga/build/simulation_pass.json]} {error "RTL/export changed after core simulation: $file"}
}
foreach file [list fpga/rtl/classifier_core.sv fpga/rtl/uart.sv fpga/rtl/basys3_classifier.sv fpga/sim/tb_uart.sv] {
    if {[file mtime $file]>[file mtime fpga/build/uart_simulation_pass.json]} {error "RTL changed after UART simulation: $file"}
}
set fd [open fpga/export/hardware_identity.json r];set identity [read $fd];close $fd
regexp {"model_id_hex": "([0-9a-f]+)"} $identity all model_id
foreach marker {simulation_pass.json uart_simulation_pass.json} {
    set fd [open fpga/build/$marker r];set simulation [read $fd];close $fd
    if {![string match "*$model_id*" $simulation]} {error "Simulation/export identity mismatch in $marker"}
}
# "resume" (-tclargs resume) reuses fpga/build/post_synth.dcp, but only if that
# checkpoint is newer than every RTL/export source and both simulation markers.
set resume [expr {[lsearch -exact $argv resume]>=0}]
if {$resume} {
    if {![file exists fpga/build/post_synth.dcp]} {error "No synthesized checkpoint to resume from"}
    foreach file [list fpga/rtl/classifier_core.sv fpga/rtl/uart.sv fpga/rtl/basys3_classifier.sv fpga/export/layers.svh fpga/export/hardware_id.svh fpga/export/weights.mem fpga/export/parameters.mem fpga/export/exp.mem fpga/constraints/basys3.xdc fpga/build/simulation_pass.json fpga/build/uart_simulation_pass.json] {
        if {[file mtime $file]>[file mtime fpga/build/post_synth.dcp]} {error "Checkpoint is older than $file; run a full build"}
    }
    open_checkpoint fpga/build/post_synth.dcp
    puts "RESUMED_FROM fpga/build/post_synth.dcp"
} else {
    create_project -in_memory -part xc7a35tcpg236-1
    set_property include_dirs [list [file normalize fpga/export]] [current_fileset]
    read_verilog -sv [list fpga/rtl/classifier_core.sv fpga/rtl/uart.sv fpga/rtl/basys3_classifier.sv]
    read_xdc fpga/constraints/basys3.xdc
    synth_design -top basys3_classifier -part xc7a35tcpg236-1 -flatten_hierarchy rebuilt
    report_utilization -file fpga/build/utilization_synth.rpt
    report_utilization -hierarchical -file fpga/build/utilization_synth_hier.rpt
    write_checkpoint -force fpga/build/post_synth.dcp
}

# Memory-mapping gate: both activation banks must be block RAM, not LUTRAM.
set all_brams [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB*}]
set lutram [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAM32* || REF_NAME =~ RAM64* || REF_NAME =~ RAM128* || REF_NAME =~ RAM256* || REF_NAME =~ RAMD* || REF_NAME =~ RAMS*}]
puts "SYNTH_TOTAL_BRAM_PRIMITIVES [llength $all_brams]"
puts "SYNTH_LUTRAM_PRIMITIVES [llength $lutram]"
foreach bank {activation_a activation_b} {
    set cells [get_cells -hierarchical -quiet -filter "REF_NAME =~ RAMB* && NAME =~ *$bank*"]
    puts "SYNTH_BRAM_PRIMITIVES_$bank [llength $cells]"
    if {[llength $cells]==0} {error "Activation bank $bank did not map to block RAM; refusing implementation"}
}
if {[llength $lutram]>512} {error "Unexpected LUTRAM count [llength $lutram]; activation memory mapping is still wrong"}

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force fpga/build/post_route.dcp
report_utilization -file fpga/build/utilization.rpt
report_utilization -hierarchical -file fpga/build/utilization_hier.rpt
report_ram_utilization -file fpga/build/ram_utilization.rpt
report_timing_summary -report_unconstrained -max_paths 10 -file fpga/build/timing.rpt
report_route_status -file fpga/build/route_status.rpt
report_drc -name final_drc -file fpga/build/drc.rpt
report_methodology -file fpga/build/methodology.rpt

# Setup (max-delay) and hold (min-delay) checks on the worst routed paths.
set setup_paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1 -nworst 1]
if {[llength $setup_paths]==0 || [llength $hold_paths]==0} {error "No timed paths; refusing bitstream"}
set wns [get_property SLACK [lindex $setup_paths 0]]
set whs [get_property SLACK [lindex $hold_paths 0]]
puts "IMPLEMENTED_WNS_NS $wns"
puts "IMPLEMENTED_WHS_NS $whs"
puts "WORST_SETUP_PATH [get_property STARTPOINT_PIN [lindex $setup_paths 0]] -> [get_property ENDPOINT_PIN [lindex $setup_paths 0]]"
set unrouted [llength [get_nets -hierarchical -quiet -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == CONFLICTS}]]
puts "UNROUTED_OR_CONFLICT_NETS $unrouted"
set drc_errors [llength [get_drc_violations -quiet -name final_drc -filter {SEVERITY == Error}]]
set drc_critical [llength [get_drc_violations -quiet -name final_drc -filter {SEVERITY == "Critical Warning"}]]
puts "DRC_ERRORS $drc_errors DRC_CRITICAL_WARNINGS $drc_critical"
if {$wns<0} {error "100 MHz setup timing failed (WNS $wns ns); refusing to generate an invalid bitstream"}
if {$whs<0} {error "Hold timing failed (WHS $whs ns); refusing to generate an invalid bitstream"}
if {$unrouted>0} {error "Design is not fully routed"}
if {$drc_errors>0} {error "Routed design has DRC errors; see fpga/build/drc.rpt"}

set lut [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ LUT*}]]
set ff [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ FD*}]]
set dsp [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ DSP48*}]]
set ramb36 [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB36*}]]
set ramb18 [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB18*}]]
write_bitstream -force fpga/build/basys3_classifier.bit
set fd [open fpga/build/build_status.json w]
puts $fd "{\"model_id_hex\":\"$model_id\",\"part\":\"xc7a35tcpg236-1\",\"clock_hz\":100000000,\"wns_ns\":$wns,\"whs_ns\":$whs,\"drc_errors\":$drc_errors,\"drc_critical_warnings\":$drc_critical,\"lut_cells\":$lut,\"ff_cells\":$ff,\"dsp_cells\":$dsp,\"ramb36\":$ramb36,\"ramb18\":$ramb18,\"bitstream\":\"fpga/build/basys3_classifier.bit\"}"
close $fd
puts "BITSTREAM_READY fpga/build/basys3_classifier.bit MODEL_ID $model_id"
exit
