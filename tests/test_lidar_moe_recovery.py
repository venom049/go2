"""Exercise real Unix recovery queries and child exits without ROS or DDS."""
import json,os,socket,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'deploy/lidar_moe_20261106';sys.path.insert(0,str(PACKAGE))
from lidar_moe.transport import FEEDBACK,QUERY
from lidar_moe.sensor_service import may_restart,query_controller
class Recovery(unittest.TestCase):
 def test_restart_decision_is_closed_by_default(self):
  for reply in [None,(1,100),(2,100)]:self.assertFalse(may_restart(reply))
  for reply in [(0,0),(3,100)]:self.assertTrue(may_restart(reply))
 def exercise(self,phase,independent=False):
  with tempfile.TemporaryDirectory(prefix='g2r-') as tmp:
   root=Path(tmp);(root/'run').mkdir();(root/'config').mkdir();(root/'config/supervisor.json').write_text(json.dumps({'backend':'raw_lidar_body_imu'} if independent else {}));fake=root/'fake/lidar_moe';fake.mkdir(parents=True)
   (fake/'__init__.py').write_text('__path__.append('+repr(str(PACKAGE/'lidar_moe'))+')\n')
   (fake/'ros_bridge.py').write_text("""import argparse,pathlib,socket
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a,_=p.parse_known_args()
c=a.root/'count';n=int(c.read_text())+1 if c.exists() else 1;c.write_text(str(n));a.output.write_text('{}\\n')
if n==1:
 s=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);s.bind(str(a.root/'run/perception_client.sock'));raise SystemExit(1)
""")
   (fake/'independent_service.py').write_text((fake/'ros_bridge.py').read_text())
   server=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);server.bind(str(root/'run/perception.sock'));server.settimeout(.1);done=threading.Event();queries=[]
   def serve():
    while not done.is_set():
     try:data,peer=server.recvfrom(128)
     except socket.timeout:continue
     queries.append(data);server.sendto(FEEDBACK.pack(b'G2F2',2,time.monotonic_ns(),phase,42),peer)
   t=threading.Thread(target=serve);t.start()
   try:
    env=dict(os.environ,PYTHONPATH=str(root/'fake')+os.pathsep+str(PACKAGE));r=subprocess.run([sys.executable,'-m','lidar_moe.sensor_service','--root',str(root),'--seconds','5','--output',str(root/'sensor.jsonl')],env=env,capture_output=True,text=True,timeout=5)
   finally:done.set();t.join();server.close()
   self.assertEqual(queries,[QUERY.pack(b'G2Q2',2)])
   allowed=phase in (0,3);self.assertEqual(int((root/'count').read_text()),2 if allowed else 1)
   self.assertEqual(r.returncode,0 if allowed else 1,r.stderr)
   log=[json.loads(x) for x in (root/'sensor_service.jsonl').read_text().splitlines()]
   self.assertEqual(next(x for x in log if x['event']=='restart_decision')['permitted'],allowed)
 def test_hold_restart(self):self.exercise(3)
 def test_pre_handoff_restart(self):self.exercise(0)
 def test_active_motion_no_restart(self):self.exercise(1)
 def test_stopped_no_restart(self):self.exercise(2)
 def test_independent_group_hold_restart(self):self.exercise(3,True)
 def test_independent_group_motion_no_restart(self):self.exercise(1,True)
 def test_controller_unavailable_no_restart(self):
  with tempfile.TemporaryDirectory(prefix='g2r-') as tmp:
   root=Path(tmp);(root/'run').mkdir();self.assertIsNone(query_controller(root));self.assertFalse(list((root/'run').iterdir()))
if __name__=='__main__':unittest.main()
