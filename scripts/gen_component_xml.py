#!/usr/bin/env python3
"""
Generate a complete, Vivado-compatible IP-XACT component.xml from HLS RTL.

Vivado requires <spirit:model> with <spirit:ports> listing every port.
This script parses the top-level Verilog module and emits a standards-compliant
component.xml that passes Vivado's IP catalog validation.

Usage:
    python scripts/gen_component_xml.py <top_verilog.v> <output_dir>
"""

import re
import sys
from pathlib import Path
from datetime import datetime

# ── IP metadata ──────────────────────────────────────────────────
IP_VENDOR = "nn_accel"
IP_LIBRARY = "nn"
IP_NAME = "nn_engine"
IP_VERSION = "1.0"
IP_DESCRIPTION = "NN Accelerator Engine - HLS-generated compute datapath for neural network inference"
PART = "xc7z045ffg900-2"
FAMILY = "zynq"
CLOCK_PERIOD = "6.67"

# ── Bus interface port groupings (HLS naming conventions) ──────
# AXI4 Master (m_axi_ddr)
AXI4_MASTER_LOGICAL = [
    "AWVALID", "AWREADY", "AWADDR", "AWID", "AWLEN", "AWSIZE",
    "AWBURST", "AWLOCK", "AWCACHE", "AWPROT", "AWQOS", "AWREGION",
    "AWUSER", "WVALID", "WREADY", "WDATA", "WSTRB", "WLAST", "WID",
    "WUSER", "ARVALID", "ARREADY", "ARADDR", "ARID", "ARLEN",
    "ARSIZE", "ARBURST", "ARLOCK", "ARCACHE", "ARPROT", "ARQOS",
    "ARREGION", "ARUSER", "RVALID", "RREADY", "RDATA", "RLAST",
    "RID", "RUSER", "RRESP", "BVALID", "BREADY", "BRESP", "BID", "BUSER",
]

# AXI-Lite Slave — control register interface
AXILITE_LOGICAL = [
    "AWVALID", "AWREADY", "AWADDR", "WVALID", "WREADY", "WDATA",
    "WSTRB", "ARVALID", "ARREADY", "ARADDR", "RVALID", "RREADY",
    "RDATA", "RRESP", "BVALID", "BREADY", "BRESP",
]

# ── XML templates ───────────────────────────────────────────────
XML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<!--
  NN Accelerator Engine - Vivado IP-XACT component (auto-generated)
  Generated: {timestamp}
  Part: {part}
  Clock: ap_clk @ {clock_period} ns ({clock_mhz:.0f} MHz)
-->
<spirit:component xmlns:xilinx="http://www.xilinx.com"
                  xmlns:spirit="http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  xilinx:version="1.0">
  <spirit:vendor>{vendor}</spirit:vendor>
  <spirit:library>{library}</spirit:library>
  <spirit:name>{name}</spirit:name>
  <spirit:version>{version}</spirit:version>
"""

XML_FOOTER = """</spirit:component>"""


def parse_verilog_ports(vfile: str) -> list[dict]:
    """Extract port name, direction, width from Verilog module header."""
    with open(vfile, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Find module declaration
    m = re.search(r"module\s+\w+\s*\((.*?)\);", content, re.DOTALL)
    if not m:
        raise ValueError(f"No module declaration found in {vfile}")

    # Find parameter definitions (for width constants)
    params = {}
    for pm in re.finditer(
        r"parameter\s+(?:integer\s+)?(\w+)\s*=\s*(\d+)", content
    ):
        params[pm.group(1)] = int(pm.group(2))

    # Default widths
    default_w = {
        "C_M_AXI_DDR_PORT_ADDR_WIDTH": 32,
        "C_M_AXI_DDR_PORT_DATA_WIDTH": 64,
        "C_M_AXI_DDR_PORT_ID_WIDTH": 1,
        "C_M_AXI_DDR_PORT_AWUSER_WIDTH": 1,
        "C_M_AXI_DDR_PORT_WUSER_WIDTH": 1,
        "C_M_AXI_DDR_PORT_ARUSER_WIDTH": 1,
        "C_M_AXI_DDR_PORT_RUSER_WIDTH": 1,
        "C_M_AXI_DDR_PORT_BUSER_WIDTH": 1,
        "C_M_AXI_DDR_PORT_WSTRB_WIDTH": 8,
        "C_S_AXI_CONTROL_ADDR_WIDTH": 7,
        "C_S_AXI_CONTROL_DATA_WIDTH": 32,
        "C_S_AXI_CONTROL_WSTRB_WIDTH": 4,
        "C_S_AXI_AXILITES_ADDR_WIDTH": 7,
        "C_S_AXI_AXILITES_DATA_WIDTH": 32,
        "C_S_AXI_AXILITES_WSTRB_WIDTH": 4,
    }
    default_w.update(params)

    ports = []
    for line in content.split("\n"):
        # Match: input/output [WIDTH:0] port_name;
        # or:    input/output port_name;
        m = re.match(
            r"^\s*(input|output|inout)\s*(?:wire\s+)?(?:\[(.+?)\]\s+)?(\w+)\s*[,;]",
            line,
        )
        if m:
            direction = m.group(1)
            width_expr = m.group(2)
            port_name = m.group(3)

            width = 1
            if width_expr:
                # Try to resolve the expression, e.g. "C_M_AXI_DDR_PORT_ADDR_WIDTH - 1"
                w = width_expr.strip()
                try:
                    # Simple expressions: NAME - 1 or NAME
                    parts = w.split("-")
                    base = parts[0].strip()
                    offset = int(parts[1].strip()) if len(parts) > 1 else 0
                    if base in default_w:
                        width = default_w[base] - offset
                    else:
                        width = 1  # unknown
                except (ValueError, IndexError):
                    width = 1

            ports.append({
                "name": port_name,
                "direction": direction,
                "width": width,
            })

    return ports


def resolve_width(width: int) -> str:
    """Return the IP-XACT vector definition for a given width."""
    if width <= 1:
        return None  # scalar
    left = width - 1
    return f"{left}:0"


def build_port_xml(port: dict) -> str:
    """Build <spirit:port> element for a single port."""
    name = port["name"]
    direction = port["direction"]
    width = port["width"]

    dir_map = {"input": "in", "output": "out", "inout": "inout"}
    spirit_dir = dir_map.get(direction, "inout")

    lines = [f'    <spirit:port>']
    lines.append(f'      <spirit:name>{name}</spirit:name>')
    lines.append(f'      <spirit:wire>')
    lines.append(f'        <spirit:direction>{spirit_dir}</spirit:direction>')

    vec = resolve_width(width)
    if vec:
        lines.append(f'        <spirit:vector>')
        lines.append(f'          <spirit:left>{vec.split(":")[0]}</spirit:left>')
        lines.append(f'          <spirit:right>0</spirit:right>')
        lines.append(f'        </spirit:vector>')

    lines.append(f'      </spirit:wire>')
    lines.append(f'    </spirit:port>')
    return "\n".join(lines)


def build_bus_interface(if_name: str, if_type: str, role: str,
                        prefix: str, logical_signals: list[str],
                        addr_width: int, data_width: int) -> str:
    """Build a <spirit:busInterface> element."""
    if role == "slave":
        role_xml = f'      <spirit:slave>\n        <spirit:memoryMapRef spirit:memoryMapRef="{if_name}"/>\n      </spirit:slave>'
    else:
        role_xml = f'      <spirit:master/>'

    lines = [f'    <spirit:busInterface>']
    lines.append(f'      <spirit:name>{if_name}</spirit:name>')
    lines.append(f'      <spirit:busType spirit:vendor="xilinx.com" spirit:library="interface" spirit:name="aximm" spirit:version="1.0"/>')
    lines.append(f'      <spirit:abstractionType spirit:vendor="xilinx.com" spirit:library="interface" spirit:name="aximm_rtl" spirit:version="1.0"/>')
    lines.append(role_xml)
    lines.append(f'      <spirit:portMaps>')

    for sig in logical_signals:
        phys = f"{prefix}_{sig}"
        lines.append(f'        <spirit:portMap>')
        lines.append(f'          <spirit:logicalPort><spirit:name>{sig}</spirit:name></spirit:logicalPort>')
        lines.append(f'          <spirit:physicalPort><spirit:name>{phys}</spirit:name></spirit:physicalPort>')
        lines.append(f'        </spirit:portMap>')

    lines.append(f'      </spirit:portMaps>')
    lines.append(f'      <spirit:parameters>')
    lines.append(f'        <spirit:parameter><spirit:name>ADDR_WIDTH</spirit:name><spirit:value spirit:format="long">{addr_width}</spirit:value></spirit:parameter>')
    lines.append(f'        <spirit:parameter><spirit:name>DATA_WIDTH</spirit:name><spirit:value spirit:format="long">{data_width}</spirit:value></spirit:parameter>')
    if role == "master":
        lines.append(f'        <spirit:parameter><spirit:name>HAS_BURST</spirit:name><spirit:value spirit:format="long">1</spirit:value></spirit:parameter>')
        lines.append(f'        <spirit:parameter><spirit:name>SUPPORTS_NARROW_BURST</spirit:name><spirit:value spirit:format="long">0</spirit:value></spirit:parameter>')
    lines.append(f'      </spirit:parameters>')
    lines.append(f'    </spirit:busInterface>')
    return "\n".join(lines)


def build_memory_map(name: str, addr_width: int) -> str:
    """Build <spirit:memoryMap> for an AXI-Lite slave."""
    range_val = 2 ** addr_width
    lines = [
        f'    <spirit:memoryMap>',
        f'      <spirit:name>{name}</spirit:name>',
        f'      <spirit:addressBlock>',
        f'        <spirit:name>reg0</spirit:name>',
        f'        <spirit:baseAddress spirit:format="long">0</spirit:baseAddress>',
        f'        <spirit:range spirit:format="long">{range_val}</spirit:range>',
        f'        <spirit:width spirit:format="long">32</spirit:width>',
        f'        <spirit:usage>register</spirit:usage>',
        f'        <spirit:access>read-write</spirit:access>',
        f'      </spirit:addressBlock>',
        f'    </spirit:memoryMap>',
    ]
    return "\n".join(lines)


def build_file_sets(rtl_dir: str) -> str:
    """Build <spirit:fileSets> from files in the RTL directory."""
    rtl_path = Path(rtl_dir)
    vfiles = sorted(rtl_path.glob("*.v"))
    datfiles = sorted(rtl_path.parent.glob("data/*.dat"))

    lines = [f'  <spirit:fileSets>']
    lines.append(f'    <spirit:fileSet>')
    lines.append(f'      <spirit:name>synthesis</spirit:name>')

    for vf in vfiles:
        lines.append(f'      <spirit:file>')
        lines.append(f'        <spirit:name>hdl/{vf.name}</spirit:name>')
        lines.append(f'        <spirit:fileType>verilogSource</spirit:fileType>')
        lines.append(f'      </spirit:file>')

    for df in datfiles:
        lines.append(f'      <spirit:file>')
        lines.append(f'        <spirit:name>data/{df.name}</spirit:name>')
        lines.append(f'        <spirit:fileType>memoryInit</spirit:fileType>')
        lines.append(f'      </spirit:file>')

    lines.append(f'    </spirit:fileSet>')
    lines.append(f'  </spirit:fileSets>')
    return "\n".join(lines)


def build_vendor_extensions(family: str) -> str:
    """Build <spirit:vendorExtensions> with Xilinx-required taxonomy and metadata.

    Vivado 2018.3 REQUIRES xilinx:coreExtensions with xilinx:taxonomy (SINGULAR,
    not taxonomies plural) inside spirit:vendorExtensions to recognize and
    register the IP in its catalog. Without this block, Vivado raises
    "ip flow 19-1977, unable to read ip file component.xml".
    """
    lines = [
        f'  <spirit:vendorExtensions>',
        f'    <xilinx:coreExtensions>',
        f'      <xilinx:supportedFamilies>',
        f'        <xilinx:family>{family}</xilinx:family>',
        f'      </xilinx:supportedFamilies>',
        f'      <xilinx:taxonomy>/UserIP</xilinx:taxonomy>',
        f'      <xilinx:displayName>NN Accelerator Engine v{IP_VERSION}</xilinx:displayName>',
        f'      <xilinx:coreVersion>{IP_VERSION}.0</xilinx:coreVersion>',
        f'      <xilinx:coreInformation>',
        f'        <xilinx:coreSupported>true</xilinx:coreSupported>',
        f'      </xilinx:coreInformation>',
        f'    </xilinx:coreExtensions>',
        f'  </spirit:vendorExtensions>',
    ]
    return "\n".join(lines)


def build_views() -> str:
    """Build <spirit:views> with a synthesis view referencing the fileSet.

    Vivado requires at least one view to link file sets to the tool flow.
    The envIdentifier format is ":vivado.xilinx.com:synthesis" with the
    language specified separately via <spirit:language>.
    """
    lines = [
        f'  <spirit:views>',
        f'    <spirit:view>',
        f'      <spirit:name>xilinx_anylanguagesynthesis</spirit:name>',
        f'      <spirit:envIdentifier>:vivado.xilinx.com:synthesis</spirit:envIdentifier>',
        f'      <spirit:language>Verilog</spirit:language>',
        f'      <spirit:fileSetRef>',
        f'        <spirit:localName>synthesis</spirit:localName>',
        f'      </spirit:fileSetRef>',
        f'    </spirit:view>',
        f'  </spirit:views>',
    ]
    return "\n".join(lines)


def generate_component_xml(verilog_top: str, output_dir: str) -> None:
    """Main entry: generate a complete component.xml."""
    ports = parse_verilog_ports(verilog_top)

    if not ports:
        raise ValueError(f"No ports found in {verilog_top}")

    print(f"Parsed {len(ports)} ports from {verilog_top}")

    # Organize ports by bus interface
    axi4_ports = [p for p in ports if p["name"].startswith("m_axi_ddr_port_")]
    control_ports = [p for p in ports if p["name"].startswith("s_axi_control_")]
    axilite_ports = [p for p in ports if p["name"].startswith("s_axi_AXILiteS_")]
    other_ports = [p for p in ports if not any(
        p["name"].startswith(pref) for pref in
        ["m_axi_ddr_port_", "s_axi_control_", "s_axi_AXILiteS_"]
    )]

    print(f"  AXI4 master:   {len(axi4_ports)} ports")
    print(f"  AXI-Lite ctrl: {len(control_ports)} ports")
    print(f"  AXI-Lite ext:  {len(axilite_ports)} ports")
    print(f"  Other:         {len(other_ports)} ports ({[p['name'] for p in other_ports]})")

    clock_mhz = 1000.0 / float(CLOCK_PERIOD)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build XML
    # ── CRITICAL: IP-XACT IEEE 1685-2009 schema REQUIRES strict
    #    element ordering. Vivado validates against the schema;
    #    wrong order → "ip flow 19-1977, unable to read".
    #    Correct order per schema:
    #      vendor → library → name → version →
    #      busInterfaces → memoryMaps → model →
    #      fileSets → clockDrivers → description → vendorExtensions
    # ─────────────────────────────────────────────────────────
    parts = []
    parts.append(XML_HEADER.format(
        timestamp=timestamp,
        part=PART,
        clock_period=CLOCK_PERIOD,
        clock_mhz=clock_mhz,
        vendor=IP_VENDOR,
        library=IP_LIBRARY,
        name=IP_NAME,
        version=IP_VERSION,
    ))

    # ── 1. <spirit:busInterfaces> (BEFORE model per schema) ──────
    parts.append('  <spirit:busInterfaces>')

    # AXI4 Master (m_axi_ddr)
    parts.append(build_bus_interface(
        "m_axi_ddr", "aximm", "master",
        "m_axi_ddr_port", AXI4_MASTER_LOGICAL,
        addr_width=32, data_width=64,
    ))

    # AXI-Lite Control slave
    parts.append(build_bus_interface(
        "s_axi_control", "aximm", "slave",
        "s_axi_control", AXILITE_LOGICAL,
        addr_width=7, data_width=32,
    ))

    # AXI-Lite ext slave (if present)
    if axilite_ports:
        parts.append(build_bus_interface(
            "s_axi_AXILiteS", "aximm", "slave",
            "s_axi_AXILiteS", AXILITE_LOGICAL,
            addr_width=7, data_width=32,
        ))

    parts.append('  </spirit:busInterfaces>')

    # ── 2. <spirit:memoryMaps> (BEFORE model per schema) ────────
    parts.append('  <spirit:memoryMaps>')
    parts.append(build_memory_map("s_axi_control", 7))
    if axilite_ports:
        parts.append(build_memory_map("s_axi_AXILiteS", 7))
    parts.append('  </spirit:memoryMaps>')

    # ── 3. <spirit:model> (AFTER busInterfaces + memoryMaps) ────
    parts.append('  <spirit:model>')
    parts.append(build_views())
    parts.append('    <spirit:ports>')
    for p in ports:
        parts.append(build_port_xml(p))
    parts.append('    </spirit:ports>')
    parts.append('  </spirit:model>')

    # ── 4. <spirit:fileSets> (AFTER model per schema) ────────────
    parts.append(build_file_sets(Path(verilog_top).parent.as_posix()))

    # ── 5. <spirit:clockDrivers> ─────────────────────────────────
    parts.append('  <spirit:clockDrivers>')
    parts.append('    <spirit:clockDriver>')
    parts.append('      <spirit:clockName>ap_clk</spirit:clockName>')
    parts.append(f'      <spirit:clockPeriod spirit:format="float">{CLOCK_PERIOD}</spirit:clockPeriod>')
    parts.append('    </spirit:clockDriver>')
    parts.append('  </spirit:clockDrivers>')

    # ── 6. <spirit:description> (AFTER fileSets per schema) ──────
    parts.append(f'  <spirit:description>{IP_DESCRIPTION}</spirit:description>')

    # ── 7. <spirit:vendorExtensions> (LAST per schema) ───────────
    parts.append(build_vendor_extensions(FAMILY))

    # ── Close ────────────────────────────────────────────────────
    parts.append(XML_FOOTER)

    xml_content = "\n".join(parts)

    # Write
    out_path = Path(output_dir) / "component.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_content, encoding="utf-8")
    print(f"\nGenerated: {out_path} ({len(xml_content)} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    generate_component_xml(sys.argv[1], sys.argv[2])
