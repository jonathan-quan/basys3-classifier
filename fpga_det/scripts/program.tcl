set root [file normalize [file join [file dirname [info script]] ../..]]
cd $root
file delete -force fpga_det/build/program_status.json
if {![file exists fpga_det/build/build_status.json] || ![file exists fpga_det/build/basys3_detector.bit]} {error "No successful implemented bitstream"}
if {![file exists fpga_det/build/uart_simulation_pass.json]} {error "UART integration verification is missing"}
foreach file [list fpga_det/rtl/detector_core.sv fpga/rtl/uart.sv fpga_det/rtl/basys3_detector.sv] {
    if {[file mtime $file]>[file mtime fpga_det/build/basys3_detector.bit]} {error "Source changed after bitstream build"}
}
open_hw_manager
connect_hw_server -allow_non_jtag
set targets [get_hw_targets -quiet *210183BF7BDEA]
if {[llength $targets]!=1} {error "Expected the user's Basys 3 serial 210183BF7BDEA; target missing/ambiguous: [get_hw_targets -quiet]"}
open_hw_target [lindex $targets 0]
set devices [get_hw_devices -quiet xc7a35t_0]
if {[llength $devices]!=1} {error "Expected exactly one XC7A35T device"}
set device [lindex $devices 0]
current_hw_device $device
# Volatile JTAG configuration only. No configuration-memory (SPI flash) device is created or touched.
set_property PROGRAM.FILE [file normalize fpga_det/build/basys3_detector.bit] $device
program_hw_devices $device
refresh_hw_device $device
puts "PROGRAMMED_DEVICE $device PART [get_property PART $device] TARGET [current_hw_target]"
# Diagnostics only; a missing property must not be confused with a programming failure.
set done_pin unknown
if {[catch {set done_pin [get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $device]} message]} {puts "DONE_PIN_DIAGNOSTIC_UNAVAILABLE $message"}
if {[catch {puts "CONFIG_STATUS [get_property REGISTER.CONFIG_STATUS $device]"} message]} {puts "CONFIG_STATUS_DIAGNOSTIC_UNAVAILABLE $message"}
puts "DONE_PIN $done_pin"
set fd [open fpga_det/build/program_status.json w]
puts $fd "{\"programmed\":true,\"method\":\"volatile JTAG (program_hw_devices)\",\"spi_flash_touched\":false,\"target\":\"[current_hw_target]\",\"device\":\"$device\",\"done_pin\":\"$done_pin\",\"bitstream\":\"fpga_det/build/basys3_detector.bit\"}"
close $fd
close_hw_target
disconnect_hw_server
close_hw_manager
puts "PROGRAMMING_COMPLETE"
exit
