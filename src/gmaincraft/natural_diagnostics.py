"""Non-policy action/target diagnostics for real unscripted Minecraft runs.

These counters NEVER enter a neural observation or a reward. They describe
what was attempted (including whether a vanilla crosshair hit wood), rather
than pretending a surface sighting means successful mining or inventory gain.
"""
from .natural_online_policy import HEAD_SIZES, wood_id


class NaturalActionAudit:
    def __init__(self):
        self.action_counts = [[0] * n for n in HEAD_SIZES]
        self.crosshair_block_decisions = 0
        self.crosshair_wood_decisions = 0
        self.attack_decisions = 0
        self.attack_on_wood_decisions = 0
        self.attack_without_look_decisions = 0
        self.wood_attack_without_look_decisions = 0
        self.mining_active_decisions = 0
        self.mining_stage_known_decisions = 0
        self.attack_still_streak = 0
        self.max_attack_still_streak = 0

    def observe(self, action, observation, crosshair_block):
        if len(action) != len(HEAD_SIZES) or any(
            type(k) is not int or not 0 <= k < HEAD_SIZES[j]
            for j, k in enumerate(action)
        ):
            raise ValueError('Invalid action audit head indices')
        for j, k in enumerate(action):
            self.action_counts[j][k] += 1
        block_id = crosshair_block.block_id if crosshair_block is not None else None
        on_wood = wood_id(block_id)
        attacking = action[2] == 1
        still = action[1] == 0
        self.crosshair_block_decisions += int(block_id is not None)
        self.crosshair_wood_decisions += int(on_wood)
        self.attack_decisions += int(attacking)
        self.attack_on_wood_decisions += int(attacking and on_wood)
        self.attack_without_look_decisions += int(attacking and still)
        self.wood_attack_without_look_decisions += int(attacking and still and on_wood)
        self.mining_active_decisions += int(observation.own[21] > .5)
        self.mining_stage_known_decisions += int(observation.own[20] > .5 and
                                                observation.own[22] >= 0)
        self.attack_still_streak = self.attack_still_streak + 1 if attacking and still else 0
        self.max_attack_still_streak = max(self.max_attack_still_streak,
                                           self.attack_still_streak)

    def summary(self):
        return dict(action_counts=[list(x) for x in self.action_counts],
                    crosshair_block_decisions=self.crosshair_block_decisions,
                    crosshair_wood_decisions=self.crosshair_wood_decisions,
                    attack_decisions=self.attack_decisions,
                    attack_on_wood_decisions=self.attack_on_wood_decisions,
                    attack_without_look_decisions=self.attack_without_look_decisions,
                    wood_attack_without_look_decisions=self.wood_attack_without_look_decisions,
                    mining_active_decisions=self.mining_active_decisions,
                    mining_stage_known_decisions=self.mining_stage_known_decisions,
                    max_attack_still_streak=self.max_attack_still_streak)


class WoodCrackProgress:
    """Bounded, one-time reward for advancing the player's OWN visible wood crack.

    Never supplies a target, action, or position to the actor. Requires the
    same vanilla crosshair target across consecutive fresh observations; an
    animation reset, missed frame or a new target cannot mint repeated credit.
    One Minecraft wood block can receive at most ten stage increments total.
    """
    def __init__(self, max_blocks=4096):
        if type(max_blocks) is not int or max_blocks < 1:
            raise ValueError('Invalid crack-progress memory bound')
        self.high_water = {}
        self.last_key = None
        self.last_stage = -1
        self.max_blocks = max_blocks

    def observe(self, observation, crosshair_block):
        stage = float(observation.own[22])
        active = observation.own[20] > .5 and observation.own[21] > .5
        key = None
        if active and stage >= 0 and crosshair_block is not None and wood_id(crosshair_block.block_id):
            key = (tuple(crosshair_block.xyz), crosshair_block.block_id)
        if key is None:
            self.last_key = None
            self.last_stage = -1
            return 0.
        stage = max(0, min(10, int(round(stage * 10))))
        previous_max = self.high_water.get(key, 0)
        increments = 0
        if key == self.last_key and stage > self.last_stage:
            increments = max(0, stage - previous_max)
        if key in self.high_water or len(self.high_water) < self.max_blocks:
            self.high_water[key] = max(previous_max, stage)
        else:
            increments = 0  # No credit for states we cannot deduplicate.
        self.last_key = key
        self.last_stage = stage
        return round(.30 * increments, 5)
