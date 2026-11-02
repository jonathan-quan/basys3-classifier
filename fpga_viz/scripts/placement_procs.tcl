# Writes the REAL placement of the open (routed) design for the chip-map visualizer.
proc tile_grid {t} {
    if {[catch {set c [get_property COLUMN $t];set r [get_property ROW $t]}]} {
        set c [get_property GRID_POINT_X $t];set r [get_property GRID_POINT_Y $t]
    }
    return "$c,$r"
}
proc export_placement {out} {
    file mkdir $out
    # Device background: every logic, block-RAM, DSP and I/O tile, with its clock region.
    set fd [open [file join $out device_tiles.csv] w]
    puts $fd "tile,type,column,row,clock_region"
    foreach t [get_tiles -filter {TYPE =~ CLBL* || TYPE == BRAM_L || TYPE == BRAM_R || TYPE == DSP_L || TYPE == DSP_R || TYPE =~ LIOB33* || TYPE =~ RIOB33*}] {
        set sites [get_sites -quiet -of_objects $t]
        set region ""
        if {[llength $sites]} {catch {set region [get_property CLOCK_REGION [lindex $sites 0]]}}
        puts $fd "$t,[get_property TYPE $t],[tile_grid $t],$region"
    }
    close $fd
    # Every placed primitive and the tile it sits in.
    set fd [open [file join $out placed_cells.csv] w]
    puts $fd "cell,ref,site,tile,column,row"
    set placed 0
    foreach c [get_cells -hierarchical -quiet -filter {IS_PRIMITIVE}] {
        set s [get_sites -quiet -of_objects $c]
        if {[llength $s]!=1} continue
        set t [get_tiles -quiet -of_objects $s]
        if {[llength $t]!=1} continue
        puts $fd "[get_property NAME $c],[get_property REF_NAME $c],$s,$t,[tile_grid $t]"
        incr placed
    }
    close $fd
    puts "PLACEMENT_EXPORTED cells=$placed dir=$out"
}
