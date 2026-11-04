"""Legacy RGB-only policy adapter used by the existing experiments.

**Not the future project's observation requirement:** see
`docs/OBSERVATION_POLICY.md`. The user wants direct game-engine memory-derived
skilled-player-knowable statistics, inventory, coordinates and eligible visible
blocks in BOTH training and gameplay. The CURRENT AgentView is RGB-only solely
to keep old experiments reproducible. Do not inject hidden world info or
administrative commands into scored actions. This is a software contract,
not a security sandbox for arbitrary Python code.
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
    """Historical RGB-only view for the original CNN baselines, not new design.

    The planned, versioned policy observation should expose bounded Minecraft
    memory-derived player-knowable fields (including own coordinates, status,
    inventory, counters, and perceptible block IDs). Do not add them here
    silently because that would invalidate comparisons with legacy models.
    """
    pixels: np.ndarray

    @classmethod
    def from_packet(cls, packet):
        rgb = np.asarray(packet.get_frame())
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or rgb.shape[0] == 0 or rgb.shape[1] == 0):
            raise ValueError('Expected nonempty rendered HxWx3 uint8 pixels (frame-free mode is memory-only)')
        safe_pixels = np.array(rgb, copy=True)
        safe_pixels.setflags(write=False)
        return cls(safe_pixels)


@dataclass(frozen=True)
class MultimodalView:
    """Versioned player-own state plus rendered pixels; no hidden target."""
    pixels: np.ndarray
    player: object  # PlayerKnowledgeV1 (late-bound to avoid MCio import)


@dataclass(frozen=True)
class MultimodalStatusView:
    """New V5-only input contract; legacy V4 MultimodalView is unchanged."""
    pixels: np.ndarray
    player: object  # PlayerKnowledgeV1
    status: object  # PlayerStatusV1


class MCioAgentBridge:
    """Turn AgentAction into synchronized MCio input, NEVER admin commands.

    MCio imports are delayed so pure policy/unit tests do not download or
    install Minecraft. The evaluator owns this bridge. Legacy experiments
    pass only .view() (RGB) to policies; a future adapter should expose
    explicitly filtered engine-derived player knowledge in train AND eval.
    """
    def __init__(self, controller, initial_packet):
        from mcio_ctrl.envs import env_util, mcio_env
        self.controller = controller
        self.packet = initial_packet
        self.input_manager = env_util.InputStateManager()
        # Private evaluator-only event counters, never included in AgentView.
        # Updated on EVERY intermediate game tick so stats deltas are not
        # lost during a multi-tick AgentAction.
        self._stats = env_util.StatsCache()
        self._stats.update_cache(initial_packet)
        self.input_map = mcio_env.INPUT_MAP
        if not ALLOWED_KEYS <= set(self.input_map):
            raise RuntimeError('MCio input map lacks expected keys')

    def view(self):
        return AgentView.from_packet(self.packet)

    def multimodal_view(self):
        """Identical opt-in RGB + memory interface for train and gameplay."""
        return MultimodalView(self.view().pixels,self.player_knowledge())

    def multimodal_status_view(self):
        """V5: RGB + own state + exact visible crosshair from patched MCio.

        This opt-in method fails closed if the mod/status flag is unavailable.
        It does NOT return raw world/chunk data or a server command channel.
        """
        return MultimodalStatusView(self.view().pixels,self.player_knowledge(),
                                    self.player_status())

    def player_knowledge(self):
        """Opt-in v1 memory observations, identical for training and eval.

        Not yet consumed by historical RGB CNNs. World objects need separate
        calibrated visibility research; no raw packet is exposed.
        """
        from gmaincraft.player_knowledge import PlayerKnowledgeV1
        return PlayerKnowledgeV1.from_packet(self.packet, stats=self._stats)

    def surface_objects(self):
        """Research-only fresh first-hit object groups; NOT in V5 model."""
        from gmaincraft.mcio_surface_objects import SurfaceObjectsV1
        return SurfaceObjectsV1.from_packet(self.packet)

    def mining_feedback(self):
        """Opt-in current vanilla break-animation stage, NOT hidden world data."""
        from gmaincraft.mcio_mining_feedback import MiningFeedbackV1
        return MiningFeedbackV1.from_packet(self.packet)

    def visible_items(self):
        """Research-only, same-action camera-visible dropped items (not a model input yet)."""
        from gmaincraft.mcio_visible_items import VisibleItemsV1
        return VisibleItemsV1.from_packet(self.packet)

    def player_status(self):
        """Opt-in patched-MCio own status + game crosshair target, not X-ray."""
        from gmaincraft.mcio_status import PlayerStatusV1
        return PlayerStatusV1.from_packet(self.packet)

    def stat_total(self, category, item):
        """Privileged Minecraft statistics; evaluator only, NOT policy input."""
        return int(self._stats[category][item])

    def apply(self, action: AgentAction, *, request_surface_objects=False,
              request_mining_feedback=False, request_visible_items=False,
              return_view=True):
        """Send normal actions and read every synchronized game tick.

        Memory-only callers can set return_view=False to avoid decoding and
        copying an unused RGB image. This does NOT stop MCio from capturing or
        transmitting a rendered frame; that requires a separate mod/protocol
        change. The default preserves legacy screenshot-policy behavior.
        """
        from mcio_ctrl import network
        if not isinstance(action, AgentAction):
            raise TypeError('Agent must return an AgentAction')
        for tick in range(action.ticks):
            held = set(action.keys)
            inputs = self.input_manager.process_action(
                {name: int(name in held) for name in self.input_map}, self.input_map)
            packet = network.ActionPacket(inputs=inputs, commands=[])
            if tick == action.ticks-1:
                requests=[]
                if request_surface_objects:
                    from gmaincraft.mcio_surface_objects import SurfaceObjectsRequest
                    requests.append(SurfaceObjectsRequest(True))
                if request_mining_feedback:
                    from gmaincraft.mcio_mining_feedback import MiningFeedbackRequest
                    requests.append(MiningFeedbackRequest(True))
                if request_visible_items:
                    from gmaincraft.mcio_visible_items import VisibleItemsRequest
                    requests.append(VisibleItemsRequest(True))
                if requests:
                    packet.options = requests
            if tick == 0 and (action.look_dx != 0 or action.look_dy != 0):
                packet.cursor_pos = [(float(self.packet.cursor_pos[0]) + action.look_dx,
                                      float(self.packet.cursor_pos[1]) + action.look_dy)]
            self.controller.send_action(packet)
            observed = self.controller.recv_observation()
            if observed is None or observed.version != network.MCIO_PROTOCOL_VERSION:
                raise RuntimeError('Missing or invalid MCio observation')
            self.packet = observed
            self._stats.update_cache(observed)
        return self.view() if return_view else None


# Only a controller/evaluator uses the privileged inventory packet to score.
# A policy cannot obtain it through AgentView.
def natural_log_count(packet):
    return sum(slot.count for slot in packet.inventory_main
               if (slot.id.startswith('minecraft:') and
                   slot.id.endswith(('_log', '_stem', '_wood', '_hyphae')) and
                   slot.count > 0))
