# NN Accelerator — IP Integration Script
#
# Adds the nn_engine IP to a Vivado project's IP repository
# and instantiates it in the current block design.
#
# Usage (Vivado Tcl console):
#   source integrate.tcl
#
# What this does:
#   1. Adds ip_release/ to Vivado's IP repository list
#   2. Reports the IP's interfaces for Block Design integration

set script_dir [file dirname [file normalize [info script]]]
set ip_repo [file join $script_dir nn_engine_1.0]

puts "============================================"
puts " NN Accelerator — IP Integration"
puts " IP repo: $ip_repo"
puts "============================================"

# ── Add IP repository ───────────────────────────────────────────
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog
puts "IP repository added: nn_accel:nn:nn_engine:1.0"

# ── Block Design integration guide ───────────────────────────────
puts ""
puts "Block Design Integration Steps:"
puts "  1. Open/Create Block Design"
puts "  2. Add IP → nn_engine (nn_accel library)"
puts "  3. Run Connection Automation, OR manually connect:"
puts ""
puts "  Port               Connect To                Notes"
puts "  ─────────────────  ────────────────────────  ─────────────────"
puts "  s_axi_control      PS M_AXI_GP0              AXI SmartConnect"
puts "  m_axi_ddr          PS S_AXI_HP0              AXI SmartConnect"
puts "  interrupt          Concat → IRQ_F2P[0]       PS interrupt"
puts "  ap_clk             PS FCLK_CLK0              150 MHz (6.67 ns)"
puts "  ap_rst_n           PS FCLK_RESET0_N          Active-low"
puts ""
puts "  4. Address Editor → assign base to s_axi_control (≥128 B)"
puts "  5. Validate Design → Generate Bitstream"
puts ""
puts "  Register Map (offset from AXI base):"
puts "    0x00 = CTRL        (bit0=START, bit1=RESET)"
puts "    0x04 = STATUS      (0=IDLE, 1=DONE)"
puts "    0x08 = INSTR_ADDR"
puts "    0x0C = WEIGHT_ADDR"
puts "    0x10 = INPUT_ADDR"
puts "    0x14 = OUTPUT_ADDR"
puts "    0x18 = WORKSPACE_ADDR"
