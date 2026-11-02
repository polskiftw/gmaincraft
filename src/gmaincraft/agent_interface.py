"""Fair pixel-only Minecraft policy boundary and normal-action adapter.

A policy sees AgentView (rendered pixels) and may return AgentAction.
Only the evaluator may read an underlying MCio observation packet for scoring.
The adapter never sends Minecraft commands, player teleportation, or world edits.
This is a software contract, not a security sandbox for arbitrary Python code.
"""
from dataclasses import dataclass
from math import isfinite

import numpy as np

# Exact vanilla MCio input names; do not accept arbitrary command strings.
ALLOWED_KEYS = frozenset((
    'W', 'A', 'S', 'D', 'SPACE', 'LEFT_SHIFT', 'LEFT_CONTROL',
    'LEFT_BUTTON', 'RIGHT_BUTTON', 'Q', 'E', 'F',
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
))


@dataclass(frozen=True)
class AgentAction:
    """Minecraft key/button holds and optional *relative mouse* movement.

    Each action lasts ticks synchronized game steps; camera delta is applied
    only on the first tick. There is deliberately no commands/cheats field.
    """
    keys: tuple[str, ...] = ()
    look_dx: float = 0.0
    look_dy: float = 0.0
    ticks: int = 1

    def __post_init__(self):
        if not isinstance(self.ticks, int) or isinstance(self.ticks, bool) or not 1 <= self.ticks <= 200:
            raise ValueError('ticks must be an integer from 1 to 200')
        if not isinstance(self.keys, tuple) or len(self.keys) != len(set(self.keys)):
            raise ValueError('keys must be a tuple of distinct input names')
        if not set(self.keys) <= ALLOWED_KEYS:
            raise ValueError('Unknown or forbidden Minecraft key')
        if not all(isinstance(v, (int, float)) and isfinite(v) and abs(v) <= 1200
                   for v in (self.look_dx, self.look_dy)):
            raise ValueError('Mouse delta must be finite and within +/-1200 pixels')


@dataclass(frozen=True)
class AgentView:
    """Exactly one thing a visual learner can inspect: rendered RGB pixels.

    Intentionally NO player coordinates, orientation, inventories, world seed,
    admin commands, block ids or target positions. Hidden/privileged state
    stays outside this policy-facing object.
    """
    pixels: np.ndarray

    @classmethod
    def from_packet(cls, packet):
        rgb = np.asarray(packet.get_frame())
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected rendered HxWx3 uint8 pixels')
        safe_pixels = np.array(rgb, copy=True)
        safe_pixels.setflags(write=False)
        return cls(safe_pixels)


class MCioAgentBridge:
    """Turn AgentAction into synchronized MCio input, NEVER admin commands.

    MCio imports are delayed so pure policy/unit tests do not download or
    install Minecraft. The evaluator owns this bridge and can inspect .packet
    for scoring; policy functions should receive only .view().
    """
    def __init__(self, controller, initial_packet):
        from mcio_ctrl.envs import env_util, mcio_env
        self.controller = controller
        self.packet = initial_packet
        self.input_manager = env_util.InputStateManager()
        self.input_map = mcio_env.INPUT_MAP
        if not ALLOWED_KEYS <= set(self.input_map):
            raise RuntimeError('MCio input map lacks expected keys')

    def view(self):
        return AgentView.from_packet(self.packet)

    def apply(self, action: AgentAction):
        from mcio_ctrl import network
        if not isinstance(action, AgentAction):
            raise TypeError('Agent must return an AgentAction')
        for tick in range(action.ticks):
            held = set(action.keys)
            inputs = self.input_manager.process_action(
                {name: int(name in held) for name in self.input_map}, self.input_map)
            packet = network.ActionPacket(inputs=inputs, commands=[])
            if tick == 0 and (action.look_dx != 0 or action.look_dy != 0):
                packet.cursor_pos = [(float(self.packet.cursor_pos[0]) + action.look_dx,
                                      float(self.packet.cursor_pos[1]) + action.look_dy)]
            self.controller.send_action(packet)
            observed = self.controller.recv_observation()
            if observed is None or observed.version != network.MCIO_PROTOCOL_VERSION:
                raise RuntimeError('Missing or invalid MCio observation')
            self.packet = observed
        return self.view()


# Only a controller/evaluator uses the privileged inventory packet to score.
# A policy cannot obtain it through AgentView.
def natural_log_count(packet):
    return sum(slot.count for slot in packet.inventory_main
               if (slot.id.startswith('minecraft:') and
                   slot.id.endswith(('_log', '_stem', '_wood', '_hyphae')) and
                   slot.count > 0))
