import unittest
from grasp_hand_execution import HandRunner


class HandRunnerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.sample = {'stamp': self.now, 'positions': [0.]*6}
        self.commands = []
        self.enabled = True
        self.runner = HandRunner('right', [.6]*6, self.commands.append,
                                 lambda: self.sample, lambda: self.enabled,
                                 clock=lambda: self.now, wall_clock=lambda: self.now)

    def update(self, positions, dt=.1):
        self.now += dt
        self.sample = {'stamp': self.now, 'positions': positions}
        self.runner.tick()

    def test_no_command_without_fresh_complete_feedback(self):
        self.now += 1
        with self.assertRaises(ValueError):
            self.runner.start()
        self.assertEqual(self.commands, [])

    def test_requires_three_new_matching_samples_after_command(self):
        self.runner.start()
        self.sample['positions'] = [.6]*6
        for _ in range(10):
            self.runner.tick()
        self.assertEqual(self.runner.state, 'running')
        self.update([.6]*6)
        for _ in range(10):
            self.runner.tick()
        self.assertEqual(self.runner.state, 'running')
        self.update([.6]*6)
        self.assertEqual(self.runner.state, 'running')
        self.update([.6]*6)
        self.assertEqual(self.runner.state, 'complete')

    def test_contact_short_of_command_does_not_prove_success(self):
        self.runner.start()
        self.update([.2]*6, dt=11.)
        self.assertEqual(self.runner.state, 'stopping')
        self.assertEqual(self.commands[-1], [.2]*6)
        for _ in range(3):
            self.update([.2]*6)
        self.assertEqual(self.runner.state, 'failed')

    def test_pause_holds_measured_hand_and_resume_needs_confirmation(self):
        self.runner.start()
        self.update([.2]*6)
        self.runner.pause()
        self.assertEqual(self.commands[-1], [.2]*6)
        with self.assertRaises(ValueError):
            self.runner.resume()
        for _ in range(3):
            self.update([.2]*6)
        self.assertEqual(self.runner.state, 'paused')
        self.runner.resume()
        self.assertEqual(self.commands[-1], [.6]*6)
        self.assertEqual(self.runner.state, 'running')

    def test_cancel_holds_without_opening_or_completing(self):
        self.runner.start()
        self.update([.2]*6)
        self.runner.stop()
        for _ in range(3):
            self.update([.2]*6)
        self.assertEqual(self.runner.state, 'stopped')
        self.assertEqual(self.commands, [[.6]*6, [.2]*6])

    def test_feedback_loss_does_not_claim_stop(self):
        self.runner.start()
        self.now += 1.
        self.runner.tick()
        self.assertEqual(self.runner.state, 'stop_unconfirmed')
        self.assertEqual(len(self.commands), 1)

    def test_disable_or_nonfinite_feedback_does_not_complete(self):
        self.runner.start()
        self.sample['positions'][0] = float('nan')
        self.runner.tick()
        self.assertEqual(self.runner.state, 'stop_unconfirmed')


if __name__ == '__main__':
    unittest.main()
