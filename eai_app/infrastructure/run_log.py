"""Append-only task events; task snapshots remain separate from execution history."""
import json
import time
import uuid
from pathlib import Path


class TaskRunLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = None
        self.run_id = None
        self.sequence = 0

    def start(self):
        self.run_id = uuid.uuid4().hex
        self.path = self.directory / (self.run_id + '.jsonl')
        self.sequence = 0

    def append(self, event, context):
        if self.path is None:
            self.start()
        row = {'version': 1, 'run_id': self.run_id, 'sequence': self.sequence,
               'event': event, 'recorded_at': time.time(), 'context': context}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(encoded+'\n')
        self.sequence += 1
