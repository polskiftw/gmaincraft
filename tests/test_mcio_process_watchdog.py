"""MCio crash watchdog simulated I/O; no GPU required or child launched."""
import pathlib
import sys
import unittest
from unittest import mock
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.mcio_process_watchdog import (WatchedControllerSync,
    MinecraftProcessExited,MinecraftObservationTimeout)
from mcio_ctrl import types
import zmq


class ProcessWatchdogTests(unittest.TestCase):
    def make_controller(self,polls,observations=()):
        obj=object.__new__(WatchedControllerSync)
        obj.launcher=mock.Mock()
        obj.launcher.poll.side_effect=list(polls)
        obj.observation_timeout=.05
        obj.connection_timeout=.05
        obj.stop_requested=lambda:False
        obj.check_mode=True
        obj._mcio_conn=mock.Mock()
        obj._mcio_conn.observation_socket.poll.return_value=zmq.POLLIN
        obj._mcio_conn.recv_observation.side_effect=list(observations)
        return obj

    def test_jvm_exited_during_observation_unblocks(self):
        ctrl=self.make_controller([139])
        with self.assertRaisesRegex(MinecraftProcessExited,'rc=139'):
            ctrl.recv_observation()
        ctrl._mcio_conn.observation_socket.poll.assert_not_called()

    def test_jvm_exited_after_initial_idle_poll(self):
        ctrl=self.make_controller([None,1])
        ctrl._mcio_conn.observation_socket.poll.return_value=0
        with self.assertRaisesRegex(MinecraftProcessExited,'rc=1'):
            ctrl.recv_observation()

    def test_nonblocking_packets_respect_game_mode(self):
        observation=mock.Mock(mode=types.MCioMode.SYNC)
        ctrl=self.make_controller([None],[observation])
        self.assertIs(ctrl.recv_observation(),observation)
        ctrl._mcio_conn.recv_observation.assert_called_once_with(block=False)

    def test_actual_timeout_never_waits_forever(self):
        ctrl=self.make_controller([None])
        ctrl.launcher.poll.side_effect=None
        ctrl.launcher.poll.return_value=None
        ctrl._mcio_conn.observation_socket.poll.return_value=0
        with self.assertRaises(MinecraftObservationTimeout):
            ctrl.recv_observation(timeout=.01)

    def test_connection_attempt_fails_on_dead_child(self):
        ctrl=self.make_controller([8])
        with self.assertRaises(MinecraftProcessExited):ctrl.wait_ready()

    def test_stop_interrupts_wait(self):
        ctrl=self.make_controller([None])
        ctrl._mcio_conn.observation_socket.poll.return_value=0
        ctrl.stop_requested=lambda:True
        with self.assertRaises(InterruptedError):ctrl.recv_observation()

if __name__=='__main__':unittest.main()
