/*****************************************************************************
 * NN Accelerator — PS Firmware Linker Script
 *
 * ZYNQ 7045 bare-metal memory map for ARM Cortex-A9.
 *
 * The firmware lives in upper DDR (above 0x3F000000) so that the lower
 * DDR region (0x00100000 – 0x3EFFFFFF) is entirely free for BIN
 * data, instructions, weights, and workspace buffers.
 *
 * Usage:
 *   arm-none-eabi-gcc -T lscript.ld ...
 *
 * Memory layout:
 *   DDR_BASE = 0x3F000000  →  .text, .rodata, .data, .bss, heap, stack
 *   Lower DDR (0x00100000) →  BIN file, workspace (managed by firmware)
 *****************************************************************************/

MEMORY
{
    DDR (rwx) : ORIGIN = 0x3F000000, LENGTH = 16M
}

STACK_SIZE  = 256K;
HEAP_SIZE   = 2M;

SECTIONS
{
    /* ── Code ──────────────────────────────────────────────────────── */
    .text : {
        KEEP(*(.vectors))        /* Vector table (if present)         */
        *(.text*)                 /* All .text sections                */
        *(.init*)                 /* C++ constructors                  */
        *(.fini*)                 /* C++ destructors                   */
        *(.rodata*)               /* Read-only data                    */
        . = ALIGN(4);
    } > DDR

    /* ── Initialised data ─────────────────────────────────────────── */
    .data : {
        __data_start = .;
        *(.data*)
        . = ALIGN(4);
        __data_end = .;
    } > DDR

    /* ── BSS (zero-initialised data) ──────────────────────────────── */
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > DDR

    /* ── Heap ─────────────────────────────────────────────────────── */
    .heap (NOLOAD) : {
        __heap_start = .;
        . = . + HEAP_SIZE;
        . = ALIGN(4);
        __heap_end = .;
    } > DDR

    /* ── Stack (grows downward from the top of DDR) ────────────────── */
    .stack (NOLOAD) : {
        __stack_end = .;
        . = . + STACK_SIZE;
        . = ALIGN(8);
        __stack_start = .;
    } > DDR

    /* ── Debug / discard ──────────────────────────────────────────── */
    /DISCARD/ : {
        *(.comment*)
        *(.note*)
        *(.debug*)
        *(.ARM.attributes*)
        *(.ARM.exidx*)
    }

    /* ── Symbols for runtime init ──────────────────────────────────── */
    __heap_size  = HEAP_SIZE;
    __stack_size = STACK_SIZE;
}

/* ── Entry point ──────────────────────────────────────────────────── */
ENTRY(main)
