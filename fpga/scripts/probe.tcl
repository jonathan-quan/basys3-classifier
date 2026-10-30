puts "VIVADO_VERSION [version -short]"
puts "BASYS3_PARTS [get_parts -quiet xc7a35tcpg236-1]"
open_hw_manager
if {[catch {connect_hw_server -allow_non_jtag} message]} {
    puts "HARDWARE_SERVER_ERROR $message"
} else {
    set targets [get_hw_targets -quiet]
    puts "HARDWARE_TARGETS $targets"
    foreach target $targets {
        puts "PROBE_TARGET $target"
        if {[catch {open_hw_target $target} message]} {
            puts "TARGET_ERROR $message"
        } else {
            foreach device [get_hw_devices -quiet] {
                puts "DEVICE $device PART [get_property PART $device]"
            }
            close_hw_target
        }
    }
    disconnect_hw_server
}
close_hw_manager
exit
