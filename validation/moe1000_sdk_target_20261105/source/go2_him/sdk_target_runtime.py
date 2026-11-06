"""Inference-only bounded command adapter. Does not publish motor commands."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort


class SDKTargetPolicy:
    def __init__(self,directory):
        directory=Path(directory)
        self.contract=json.loads((directory/'contract.json').read_text())
        self.session=ort.InferenceSession(str(directory/'command_policy.onnx'),providers=['CPUExecutionProvider'])
        mapping=[3,4,5,0,1,2,9,10,11,6,7,8]
        self.lower=np.array(self.contract['lower_rad'])[mapping]
        self.upper=np.array(self.contract['upper_rad'])[mapping]

    def infer(self,observation):
        obs=np.asarray(observation,dtype=np.float32)
        if obs.shape!=(1,225) or not np.isfinite(obs).all():
            raise ValueError('Expected finite float32 [1,225] observations')
        target,history,sdk=self.session.run(None,{'observations':obs})
        if (not all(np.isfinite(x).all() for x in (target,history,sdk))
                or np.any(sdk<self.lower) or np.any(sdk>self.upper)):
            raise FloatingPointError('Invalid command; publication must be rejected')
        return sdk[0].copy(),history[0].copy()
