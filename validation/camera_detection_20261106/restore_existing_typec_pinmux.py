"""Restore one value already specified by this robot's existing boot script.

No GPIO toggling, driver reload, USB reset, robot RPC or persistent system edits.
The original MMIO value is fsynced before the one aligned 32-bit write.
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply-existing-boot-value', action='store_true')
    ap.add_argument('--record', type=Path, required=True)
    args = ap.parse_args()
    if not args.apply_existing_boot_value:
        raise SystemExit('Explicit application flag required; use separate readonly probe otherwise')
    if os.geteuid() != 0:
        raise SystemExit('Root required for the existing boot pinmux operation')
    model = Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode()
    rc = Path('/etc/rc.local').read_bytes()
    if model != 'NVIDIA Orin Nano Developer Kit':
        raise SystemExit('Unexpected board; refusing hardware write')
    if b'busybox devmem 0x02430030 w 0x004' not in rc:
        raise SystemExit('Existing boot script does not specify the expected value')
    gpio = Path('/sys/class/gpio/PP.06')
    if (gpio/'direction').read_text().strip() != 'out' or (gpio/'value').read_text().strip() != '1':
        raise SystemExit('Unexpected GPIO state; no GPIO changes are permitted here')
    address, target, expected = 0x02430030, 0x04, 0x20
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    try:
        with mmap.mmap(fd, mmap.PAGESIZE, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE,
                       offset=address & ~(mmap.PAGESIZE - 1)) as mem:
            register = ctypes.c_uint32.from_buffer(mem, address % mmap.PAGESIZE)
            try:
                old = register.value
                if old not in (expected, target):
                    raise SystemExit('Unexpected current register value: ' + hex(old))
                record = {'time_unix_s': time.time(), 'board': model,
                          'existing_rc_local_sha256': hashlib.sha256(rc).hexdigest(),
                          'address': hex(address), 'original_value': hex(old),
                          'target_value': hex(target), 'write_performed': False,
                          'persistent_configuration_modified': False,
                          'gpio_modified': False, 'robot_motion_command_sent': False}
                # Never overwrite an earlier rollback record.
                with args.record.open('x') as stream:
                    stream.write(json.dumps(record, indent=2) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if old != target:
                    register.value = target
                    record['write_performed'] = True
                readback = register.value
                record['readback'] = hex(readback)
                if readback != target:
                    register.value = old
                    record['rollback_readback'] = hex(register.value)
                with args.record.open('w') as stream:
                    stream.write(json.dumps(record, indent=2) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(record), flush=True)
                if readback != target:
                    raise SystemExit('Write verification failed; rollback attempted')
            finally:
                del register
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
