"""Install only the independent stair controller's reviewed protection update.

Run on the robot after uploading the patch and offline tests. Never starts DDS,
releases factory control, or runs the controller with an actuation option.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/home/unitree/go2-stairs-torque1000-test-20261103/controller')
PACKAGE = Path(__file__).resolve().parent
MANIFEST = json.loads((PACKAGE / 'patch_manifest.json').read_text())
BINARY = ROOT / 'robots/go2/build/go2_ctrl'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_idle():
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            name = (item / 'comm').read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name == 'go2_ctrl':
            raise RuntimeError(f'Controller PID {item.name} exists: refusing live update')


def run(args, cwd=None):
    print('RUN', ' '.join(str(x) for x in args), flush=True)
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


require_idle()
for relative, expected in MANIFEST['unchanged_input_sha256'].items():
    if digest(ROOT / relative) != expected:
        raise RuntimeError(f'Unreviewed build input: {relative}')
for item in MANIFEST['files']:
    rel = Path(item['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise RuntimeError('Invalid relative patch path')
    source, target = PACKAGE / 'patch' / rel, ROOT / rel
    if digest(source) != item['new_sha256']:
        raise RuntimeError(f'Package hash mismatch: {rel}')
    if item['old_sha256'] is None:
        if target.exists():
            raise RuntimeError(f'Unexpected existing new header: {rel}')
    elif digest(target) != item['old_sha256']:
        raise RuntimeError(f'Onboard source changed since read-only audit: {rel}')
if digest(BINARY) != MANIFEST['old_binary_sha256']:
    raise RuntimeError('Existing executable differs from audited version')

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = ROOT.parent / 'backups' / ('before_official_limits_' + stamp)
backup.mkdir(parents=True, exist_ok=False)
for item in MANIFEST['files']:
    source = ROOT / item['path']
    if source.exists():
        target = backup / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
shutil.copy2(BINARY, backup / 'go2_ctrl')
result = {'backup': str(backup), 'controller': str(ROOT),
          'motor_commands_sent': False, 'factory_files_changed': False,
          'model_files_changed': False, 'status': 'backed_up'}
try:
    for item in MANIFEST['files']:
        shutil.copy2(PACKAGE / 'patch' / item['path'], ROOT / item['path'])
    for name in ['test_first_test_guard', 'test_go2_official_joint_limits',
                 'test_handoff_hold', 'test_stand_readiness']:
        output = PACKAGE / name
        run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
             '-O2', '-I' + str(ROOT / 'include'), PACKAGE / (name + '.cpp'),
             '-o', output])
        run([output])
    build = ROOT / 'robots/go2/build-official-limits'
    run(['cmake', '-S', ROOT / 'robots/go2', '-B', build,
         '-DCMAKE_BUILD_TYPE=Release',
         '-DUNITREE_SDK2_DIR=/home/unitree/go2_sdk2_new',
         '-DUNITREE_DDS_DIR=/usr/local'])
    run(['cmake', '--build', build, '-j2'])
    require_idle()
    for relative, expected in MANIFEST['unchanged_input_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Build input changed during compilation: {relative}')
    # Reviewed main.cpp returns here BEFORE DDS initialization or a publisher.
    probe = subprocess.run([str(build / 'go2_ctrl')], cwd=ROOT / 'robots/go2',
                           capture_output=True, text=True, timeout=15, check=True)
    if 'FIRST_TEST_DRY_RUN: no DDS, no publisher, no motor commands.' not in probe.stdout:
        raise RuntimeError('Expected non-actuating dry-run marker absent')
    (PACKAGE / 'dry_run.txt').write_text(probe.stdout + probe.stderr)
    require_idle()
    temporary = BINARY.with_name('go2_ctrl.official-limits.tmp')
    shutil.copy2(build / 'go2_ctrl', temporary)
    os.replace(temporary, BINARY)
    result.update(status='installed_not_started', binary=str(BINARY),
                  binary_sha256=digest(BINARY),
                  files={item['path']: digest(ROOT / item['path'])
                         for item in MANIFEST['files']},
                  offline_tests_passed=True, full_arm64_build_passed=True,
                  no_dds_dry_run_passed=True)
except BaseException:
    for item in MANIFEST['files']:
        original = backup / item['path']
        target = ROOT / item['path']
        if original.exists():
            shutil.copy2(original, target)
        elif item['old_sha256'] is None and target.exists():
            target.unlink()
    shutil.copy2(backup / 'go2_ctrl', BINARY)
    result['status'] = 'failed_rolled_back'
    (PACKAGE / 'installation.json').write_text(json.dumps(result, indent=2) + '\n')
    raise
(PACKAGE / 'installation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
