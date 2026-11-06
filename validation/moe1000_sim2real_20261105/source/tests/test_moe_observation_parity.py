"""Numerically compare native observation history to the deployment YAML contract."""
from collections import deque
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT/'_exports/go2-moe1000-20261104/params/deploy.json').read_text())
rng = np.random.default_rng(42)
rows, expected = [], []
mapping = cfg['joint_ids_map']
histories = {name: deque(maxlen=5) for name in cfg['observations']}
for index in range(200):
    q = rng.normal(0., .2, 12)
    q[np.array(mapping)] += np.asarray(cfg['default_joint_pos'])
    dq, gyro, previous = rng.normal(0., 3., 12), rng.normal(0., .3, 3), rng.normal(0., .2, 12).astype('float32')
    quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    vx, reset = (.6 if index % 2 else 0.), index % 17 == 0
    down = np.array([0., 0., -1.]); cross = np.cross(quat[1:], down)
    gravity = down + 2*(-quat[0]*cross + np.cross(quat[1:], cross))
    raw = [gyro, gravity, np.array([vx,0.,0.]), q[mapping]-cfg['default_joint_pos'], dq[mapping], previous]
    chunks = []
    for (name, config), value in zip(cfg['observations'].items(), raw):
        if reset: histories[name].clear()
        if not histories[name]: histories[name].extend([value.copy() for _ in range(5)])
        else: histories[name].append(value.copy())
        chunks.extend(np.clip(v,*config['clip'])*config['scale'] for v in histories[name])
    expected.append(np.concatenate(chunks).astype('float32'))
    rows.append(np.r_[q,dq,quat,gyro,previous,vx,float(reset)])
with tempfile.TemporaryDirectory() as directory:
    binary = str(Path(directory)/'oracle')
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'deploy/moe1000_controller/overlay/include'),str(ROOT/'tests/moe_observation_oracle.cpp'),'-o',binary],check=True)
    output = subprocess.run([binary],input=np.asarray(rows,dtype='float64').tobytes(),stdout=subprocess.PIPE,check=True).stdout
actual = np.frombuffer(output,dtype='float32').reshape(200,225)
expected = np.asarray(expected)
np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)
print(json.dumps({'passed':True,'frames':200,'history_resets':12,'max_abs_error':float(np.abs(actual-expected).max())}))
