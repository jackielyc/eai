import json
import tempfile
import unittest
from pathlib import Path

from grasp_run_log import TaskRunLog


class RunLogTests(unittest.TestCase):
    def test_append_preserves_prior_events_and_separates_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            log = TaskRunLog(directory)
            log.append('start', {'mode': 'simulation', 'observation_id': 'old-image'})
            first_path = log.path
            log.append('operator_confirmed', {'step': 2})
            rows = [json.loads(line) for line in first_path.read_text().splitlines()]
            self.assertEqual([row['sequence'] for row in rows], [0, 1])
            self.assertEqual(rows[0]['run_id'], rows[1]['run_id'])
            log.start()
            log.append('start', {'mode': 'hardware'})
            self.assertNotEqual(first_path, log.path)
            self.assertEqual(len(list(Path(directory).glob('*.jsonl'))), 2)

    def test_nonfinite_event_does_not_corrupt_previous_record(self):
        with tempfile.TemporaryDirectory() as directory:
            log = TaskRunLog(directory)
            log.append('valid', {})
            with self.assertRaises(ValueError):
                log.append('invalid', {'force': float('nan')})
            self.assertEqual(len(log.path.read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
