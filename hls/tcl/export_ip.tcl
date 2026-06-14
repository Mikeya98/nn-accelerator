# Vivado HLS — Export IP Catalog
#
# Run this script from within the Vivado HLS project to generate
# a proper IP-XACT IP that Vivado can read without errors.
#
# Usage (HLS command line):
#   vivado_hls -f hls/tcl/export_ip.tcl
#
# Or in the HLS GUI: open the solution, paste the command below
# into the Tcl Console:
#   export_design -format ip_catalog
#
# This produces a complete IP-XACT component (with valid component.xml)
# in solution1/impl/ip/ — use THAT directory as the IP repository in
# Vivado, NOT the hand-written ip_release/ directory.

# Open the existing HLS project
open_project hls/nn_accelerator/nn_accelerator.xpr
open_solution solution1

# Export as IP Catalog
#   -format ip_catalog  → full IP-XACT component with valid component.xml
#   -evaluate verilog    → use the synthesis results (already done)
export_design -format ip_catalog

puts ""
puts "=============================================="
puts " IP exported to: solution1/impl/ip/"
puts ""
puts " In Vivado: Settings → IP → Repository → Add:"
puts "   hls/nn_accelerator/solution1/impl/ip/"
puts "=============================================="
