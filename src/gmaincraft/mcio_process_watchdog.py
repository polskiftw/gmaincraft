"""Bounded synchronous MCio observation read, including JVM death detection.

MCio 1.6.1 ControllerSync.recv_observation intentionally ignores its timeout
argument. If Java/NVIDIA aborts between worlds, its ZeroMQ polling loop can
wait forever and hold the world's exclusive PPO checkpoint file lock. This
watchdog uses the SAME native MCio socket and packet protocol, but limits
connection and observation waits while checking the exact launched JVM.
It does not change gameplay, input actions, game state, or learned inputs.
"""
import time

from mcio_ctrl import controller,types
import zmq


class MinecraftProcessExited(RuntimeError):
    """The exact integrated-server client exited while observations were due."""


class MinecraftObservationTimeout(TimeoutError):
    """Neither the server nor the network produced a new observation."""


class WatchedControllerSync(controller.ControllerSync):
    def __init__(self,*,launcher,connection_timeout=45.,observation_timeout=30.,
                 stop_requested=None,**kwargs):
        if not 1<=connection_timeout<=180 or not 1<=observation_timeout<=180:
            raise ValueError('Invalid observation/connection watchdog limit')
        self.launcher=launcher
        self.connection_timeout=float(connection_timeout)
        self.observation_timeout=float(observation_timeout)
        self.stop_requested=stop_requested or (lambda:False)
        # Do not block in MCio's uninterruptible wait for connections.
        # Its two monitor events are checked with a separate process-aware
        # deadline here, including startup JVM failures before any packets.
        super().__init__(wait_for_connection=False,**kwargs)
        try:
            self.wait_ready()
        except BaseException:
            self.close()
            raise

    def _check_process(self,phase):
        rc=self.launcher.poll()
        if rc is not None:
            raise MinecraftProcessExited(
                f'Minecraft Java child exited rc={rc} during {phase} '
                'without a valid MCio observation. Exit code 0 is an '
                'unexpected clean JVM exit, NOT proof of a GPU crash; '
                'nonzero/negative return codes need the actual JVM/driver '
                'log to identify their cause. No uncommitted PPO episode '
                'will be credited.')

    def wait_ready(self):
        until=time.monotonic()+self.connection_timeout
        while True:
            self._check_process('MCio connection')
            conn=self._mcio_conn
            if conn.action_connected.is_set() and conn.observation_connected.is_set():
                return
            if self.stop_requested():
                raise InterruptedError('Stop requested while Minecraft was launching')
            if time.monotonic()>=until:
                raise MinecraftObservationTimeout(
                    f'Minecraft did not establish MCio action+observation sockets '
                    f'within {self.connection_timeout:.0f}s')
            time.sleep(.03)

    def recv_observation(self,block=True,timeout=None):
        """Compatibility with ControllerSync, but *really* enforce timeout."""
        if not block:
            raise ValueError('Synchronous MCio requires blocking packet ordering')
        limit=self.observation_timeout if timeout is None else float(timeout)
        if not 0<limit<=180:
            raise ValueError('Invalid observation deadline')
        deadline=time.monotonic()+limit
        while True:
            self._check_process('observation wait')
            # Poll the native ZMQ descriptor: immediately returns when data
            # arrives (no per-tick 25ms latency), and releases after 25ms so
            # we can detect a dead JVM or an interrupted run.
            available=self._mcio_conn.observation_socket.poll(25,zmq.POLLIN)
            if available:
                obs=self._mcio_conn.recv_observation(block=False)
                if obs is None:
                    continue
                if obs.mode!=types.MCioMode.SYNC:
                    raise RuntimeError('Wrong MCio mode in fresh survival observation')
                self.check_mode=False
                return obs
            if self.stop_requested():
                raise InterruptedError('Stop requested while waiting for MCio observation')
            if time.monotonic()>=deadline:
                raise MinecraftObservationTimeout(
                    f'No MCio observation for {limit:.1f}s while Java is '
                    'still running; refusing to hang the training lock forever')
