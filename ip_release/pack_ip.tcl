# pack_ip.tcl — Package HLS RTL as Vivado IP (minimal, reliable)
#
# USAGE (Vivado Tcl Console):
#   source E:/work/nn_ip/ip_release/pack_ip.tcl

set script_dir [file dirname [file normalize [info script]]]
set rtl_dir   [file join $script_dir nn_engine_1.0 hdl]
set dat_dir   [file join $script_dir nn_engine_1.0 data]
set out_dir   [file join $script_dir nn_engine_1.0]
set tmp_proj  [file join $script_dir .. _tmp_pack]

puts "=== NN Engine IP Packager ==="

# Clean
if {[file exists $tmp_proj]} { file delete -force $tmp_proj }
catch { file delete [file join $out_dir component.xml] }
catch { file delete -force [file join $out_dir xgui] }

# Create project and add sources
create_project -force _tmp_pack $tmp_proj -part xc7z045ffg900-2
add_files -norecurse [glob -nocomplain -dir $rtl_dir *.v]
add_files -norecurse [glob -nocomplain -dir $dat_dir *.dat]
set_property top nn_engine [current_fileset]
set_property source_mgmt_mode None [current_project]

# Package as IP
ipx::package_project \
    -root_dir $out_dir \
    -vendor nn_accel \
    -library nn \
    -taxonomy /UserIP \
    -import_files \
    -force

# Save and exit
ipx::save_core [ipx::current_core]
close_project -delete

set xml [file join $out_dir component.xml]
puts "DONE: $xml ([file size $xml] bytes)"
puts "IP repo path: [file normalize [file join $out_dir ..]]"
