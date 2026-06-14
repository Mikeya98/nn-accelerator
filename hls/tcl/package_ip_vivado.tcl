# Vivado IP Packager — packages HLS-generated RTL into a valid IP-XACT IP.
#
# Usage (command line):
#   vivado -mode batch -source hls/tcl/package_ip_vivado.tcl -tclargs <rtl_dir> <output_ip_dir>
#
# This script creates a temporary Vivado project, adds the HLS-generated
# Verilog/SystemVerilog files, and runs ipx::package_project to produce a
# complete IP-XACT component.xml that passes Vivado's IP catalog validation.
#
# Why this is needed:
#   The hand-written component.xml in ip_release/ is missing the <spirit:model>
#   and <spirit:ports> sections required by the IP-XACT IEEE 1685 spec.
#   Vivado rejects it with "ip flow 19-1977".  This script generates a
#   standards-compliant component.xml automatically.

proc package_nn_ip {rtl_dir output_ip_dir} {
    puts "================================================"
    puts " NN Accelerator IP Packaging (Vivado ipx flow)"
    puts "================================================"
    puts "  RTL dir:    $rtl_dir"
    puts "  Output dir: $output_ip_dir"
    puts ""

    # ── Step 1: Create temporary Vivado project ──────────────────
    set tmp_proj "tmp_nn_ip_proj"
    if {[file exists $tmp_proj]} {
        file delete -force $tmp_proj
    }

    create_project -in_memory -part xc7z045ffg900-2
    puts "Created in-memory project (part: xc7z045ffg900-2)"

    # ── Step 2: Add all HLS RTL source files ─────────────────────
    set verilog_files [glob -nocomplain -dir $rtl_dir *.v]
    if {[llength $verilog_files] == 0} {
        error "No Verilog files found in $rtl_dir"
    }
    puts "Found [llength $verilog_files] Verilog source files"

    # Add files; HLS may generate SV files with .v extension
    foreach f [lsort $verilog_files] {
        if {[string match "*_ip.tcl" $f]} { continue }
        set tail [file tail $f]
        puts "  Adding: $tail"
        add_files -norecurse $f
    }

    # Also add .dat ROM init files if present
    set dat_dir [file join [file dirname $rtl_dir] ".."]
    set data_dir_candidates [list \
        [file join $rtl_dir "../data"] \
        [file join $dat_dir "data"] \
    ]
    foreach d $data_dir_candidates {
        if {[file exists $d]} {
            set dat_files [glob -nocomplain -dir $d *.dat]
            foreach df $dat_files {
                puts "  Adding data: [file tail $df]"
                add_files -norecurse $df
            }
        }
    }

    # ── Step 3: Set top module and properties ────────────────────
    set_property top nn_engine [current_fileset]
    set_property source_mgmt_mode None [current_project]

    # ── Step 4: Define AXI interfaces via IP-XACT properties ─────
    # These must match the RTL port names from HLS.
    # HLS generates AXI-Lite slave on 's_axi_control' and
    # AXI4 master on 'm_axi_ddr'.
    #
    # The interface names follow HLS naming convention:
    #   s_axi_control_*  → AXI-Lite slave
    #   m_axi_ddr_*      → AXI4 master

    # ── Step 5: Package as IP ────────────────────────────────────
    puts ""
    puts "Packaging IP with ipx::package_project ..."

    # Set IP repository path
    set ip_repo [file normalize $output_ip_dir]

    # Package as IP
    ipx::package_project \
        -root_dir $ip_repo \
        -vendor nn_accel \
        -library nn \
        -taxonomy /UserIP \
        -import_files \
        -set_current false

    # Open the generated component
    set ip_name "nn_engine"
    set component [ipx::current_core]

    puts "Component created: $component"

    # ── Step 6: Configure AXI bus interfaces ─────────────────────
    puts "Configuring bus interfaces ..."

    # Map the HLS-generated AXI-Lite slave port group
    # HLS suffixes: _AWVALID, _AWREADY, _WVALID, _WREADY, etc.
    set slave_if "s_axi_control"
    set master_if "m_axi_ddr"

    # Configure s_axi_control as AXI4-Lite slave
    ipx::add_bus_interface $slave_if $component
    ipx::associate_bus_interface -busif $slave_if -clock ap_clk $component
    ipx::associate_bus_interface -busif $slave_if -reset ap_rst_n $component

    # Configure m_axi_ddr as AXI4 master
    ipx::add_bus_interface $master_if $component
    ipx::associate_bus_interface -busif $master_if -clock ap_clk $component
    ipx::associate_bus_interface -busif $master_if -reset ap_rst_n $component

    # ── Step 7: Configure address space ──────────────────────────
    puts "Configuring address spaces ..."

    # Create memory map for AXI-Lite slave
    set mm [ipx::add_memory_map "s_axi_control" $component]
    ipx::add_address_block "reg0" $mm
    ipx::associate_memory_map -busif $slave_if $mm

    # ── Step 8: Close component and finalize ─────────────────────
    ipx::check_integrity $component
    ipx::save_core $component
    ipx::unload_core $component

    puts ""
    puts "================================================"
    puts " IP packaged successfully!"
    puts " Output: [file normalize $output_ip_dir]"
    puts ""
    puts " To use in Vivado Block Design:"
    puts "   1. Settings → IP → Repository → Add $ip_repo"
    puts "   2. Add 'nn_engine' to your block design"
    puts "   3. Run Connection Automation"
    puts "================================================"
}

# ── Entry point ──────────────────────────────────────────────────
if {[llength $argv] >= 2} {
    package_nn_ip [lindex $argv 0] [lindex $argv 1]
} elseif {[llength $argv] == 1 && [lindex $argv 0] eq "--help"} {
    puts "Usage: vivado -mode batch -source package_ip_vivado.tcl -- <rtl_dir> <output_ip_dir>"
    puts ""
    puts "  rtl_dir        Directory containing HLS-generated .v files"
    puts "  output_ip_dir  Where to write the packaged IP"
    puts ""
    puts "Example:"
    puts "  vivado -mode batch -source hls/tcl/package_ip_vivado.tcl -- \\"
    puts "    hls/nn_accelerator/solution1/syn/verilog/ \\"
    puts "    ip_release/nn_engine_1.0"
} else {
    puts "ERROR: Need rtl_dir and output_ip_dir arguments."
    puts "Usage: vivado -mode batch -source package_ip_vivado.tcl -- <rtl_dir> <output_ip_dir>"
    exit 1
}
