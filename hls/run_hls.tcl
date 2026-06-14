# NN Accelerator — HLS Synthesis Script
#
# Vivado HLS 2018.3 compatible — single-file architecture.
# All compute functions live in nn_engine.cpp (same translation unit)
# to avoid the Pointer Array Geometry crash caused by passing m_axi
# pointer across module boundaries.
#
# Usage:
#   vivado_hls -f run_hls.tcl
#
# Options:
#   set DO_CSIM=1  → run C simulation before synthesis
#   set EXPORT_IP=1 → export IP catalog (may have 2018.3 packager bug)

set PROJECT_NAME   nn_accelerator
set TOP_FUNCTION   nn_engine
set CLOCK_PERIOD   6.67     ;# 150 MHz
set PART           xc7z045ffg900-2

# ── Open / create project ──────────────────────────────────────────
open_project -reset ${PROJECT_NAME}

# Single source file with all compute functions
add_files src/nn_engine.cpp

# Testbench (only needed for C simulation)
add_files -tb tb/nn_engine_tb.cpp

# ── Top function ───────────────────────────────────────────────────
set_top ${TOP_FUNCTION}

# ── Solution ──────────────────────────────────────────────────────
open_solution -reset solution1
set_part ${PART}
create_clock -period ${CLOCK_PERIOD}

# Reduce optimization pressure for Vivado 2018.3 stability
config_compile -pipeline_loops 0

# ── C Simulation (optional) ────────────────────────────────────────
if { [info exists ::env(DO_CSIM)] && $::env(DO_CSIM) } {
    set bin_path [file normalize "model.bin"]
    csim_design -clean -argv $bin_path
}

# ── Synthesis ──────────────────────────────────────────────────────
csynth_design

puts ""
puts "============================================"
puts " Synthesis complete!"
puts " RTL generated in:"
puts "   ${PROJECT_NAME}/solution1/syn/[format verilog|vhdl|systemc]"
puts "============================================"

# ── Export IP (optional — may have Vivado 2018.3 packager bug) ────
if { [info exists ::env(EXPORT_IP)] && $::env(EXPORT_IP) } {
    export_design \
        -format ip_catalog \
        -vendor  "nn_accel" \
        -library "nn" \
        -version "1.0" \
        -display_name "NN_Accelerator_v1.0"
    puts "IP exported to: ${PROJECT_NAME}/solution1/impl/ip"
}
