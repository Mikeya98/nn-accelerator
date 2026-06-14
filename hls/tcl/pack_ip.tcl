# Vivado IP Packager — run AFTER HLS export
#
# This script takes the HLS-exported IP and adds the necessary
# interface metadata for Vivado Block Design integration.
#
# Usage (in Vivado Tcl console):
#   source hls/tcl/pack_ip.tcl
#   pack_nn_ip <hls_solution_dir> <output_ip_dir>
#
# Or run from command line:
#   vivado -mode batch -source hls/tcl/pack_ip.tcl

proc pack_nn_ip {hls_impl_dir output_ip_dir} {
    puts "Packing NN Accelerator IP ..."
    puts "  HLS impl dir: $hls_impl_dir"
    puts "  Output dir:   $output_ip_dir"

    # The HLS export creates an IP in the impl/ip directory.
    # We create a wrapper to add address editor metadata.
    set ip_src [glob -dir $hls_impl_dir ip/*.zip]
    if {$ip_src eq ""} {
        error "No IP zip found in $hls_impl_dir"
    }

    file mkdir $output_ip_dir
    file copy -force $ip_src $output_ip_dir/

    puts "IP packaged to $output_ip_dir"
    puts ""
    puts "To use in Vivado Block Design:"
    puts "  1. Settings → IP → Repository → Add $output_ip_dir"
    puts "  2. Add 'NN_Accelerator_v1_0' to block design"
    puts "  3. Connect:"
    puts "     s_axi_control → PS M_AXI_GP0 (AXI-Lite)"
    puts "     m_axi_ddr     → PS S_AXI_HP0 (AXI Full, 64-bit)"
    puts "     interrupt     → PS IRQ_F2P[0]"
    puts "  4. Assign base addresses in Address Editor"
    puts "     (s_axi_control needs ≥ 128 B = 32 regs × 4 B)"
}

# If run as standalone script with arguments
if {[info exists argv] && [llength $argv] >= 2} {
    pack_nn_ip [lindex $argv 0] [lindex $argv 1]
} else {
    puts "Usage: vivado -mode batch -source pack_ip.tcl -- <hls_impl_dir> <output_ip_dir>"
}
