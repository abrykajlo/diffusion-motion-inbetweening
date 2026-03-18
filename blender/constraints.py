"""
Per-frame bone constraint storage, serialised as JSON in a scene property.
"""

import json

CONSTRAINTS_KEY = "dmi_constraints"

class Constraints:
    def __init__(self, scene):
        self.scene = scene
        self.load()

    def __len__(self):
        return len(self.data)

    def save(self):
        self.scene[CONSTRAINTS_KEY] = json.dumps(self.data)

    def load(self):
        raw = self.scene.get(CONSTRAINTS_KEY, "{}")
        if isinstance(raw, str):
            try:
                self.data = json.loads(raw)
                return
            except json.JSONDecodeError:
                pass
        self.data = {}

    def set(self, frame: int, bone: str):
        key = str(frame)
        if key not in self.data:
            self.data[key] = {}
        self.data[key][bone] = True

    def has(self, frame: int, bone: str) -> bool:
        key = str(frame)
        return key in self.data and bone in self.data[key] and self.data[key][bone]

    def remove(self, frame: int, bone: str):
        key = str(frame)
        if key in self.data and bone in self.data[key]:
            del self.data[key][bone]
            if not self.data[key]:
                del self.data[key]

    def clear(self):
        self.data = {}
