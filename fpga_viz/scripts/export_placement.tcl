# Usage: vivado -mode batch -source fpga_viz/scripts/export_placement.tcl -tclargs <post_route.dcp> <output_dir>
source [file join [file dirname [info script]] placement_procs.tcl]
open_checkpoint [file normalize [lindex $argv 0]]
export_placement [file normalize [lindex $argv 1]]
exit
