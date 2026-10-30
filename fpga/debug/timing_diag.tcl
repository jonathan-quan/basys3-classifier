# Diagnostic only: summarise every failing setup endpoint in the routed checkpoint.
set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
open_checkpoint fpga/build/post_route.dcp
set fd [open fpga/logs/timing_diag.txt w]
set paths [get_timing_paths -delay_type max -max_paths 2000 -nworst 1 -unique_pins -slack_lesser_than 0]
puts $fd "FAILING_PATHS [llength $paths]"
foreach p $paths {
    puts $fd "[get_property SLACK $p] [get_property LOGIC_LEVELS $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
}
puts $fd "NEAR_CRITICAL (slack < 1.5ns) by endpoint:"
foreach p [get_timing_paths -delay_type max -max_paths 3000 -nworst 1 -unique_pins -slack_lesser_than 1.5] {
    puts $fd "N [get_property SLACK $p] [get_property LOGIC_LEVELS $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
}
close $fd
puts "TIMING_DIAG_DONE"
exit
