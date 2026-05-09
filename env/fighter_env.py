"""
fighter_env.py
Custom 2D fighting game environment built with Pygame.

Controls (human player):
  A / D        = Walk left / right
  J            = Punch
  K            = Kick
  L            = Block
  Space        = Jump
  S            = Crouch
  U            = Fireball (Special A)
  I            = Uppercut (Special B)
"""

import pygame
import numpy as np
import math

# ── Constants ────────────────────────────────────────────────────────────────
W, H          = 900, 500
FPS           = 60
GROUND_Y      = 380
ROUND_TIME    = 60
GRAVITY       = 0.6
JUMP_VEL      = -14
WALK_SPEED    = 4.0
ARENA_LEFT    = 40
ARENA_RIGHT   = W - 90

IDLE      = 0
PUNCH     = 1
KICK      = 2
BLOCK     = 3
JUMP      = 4
CROUCH    = 5
SPECIAL_A = 6
SPECIAL_B = 7
N_ACTIONS = 8

WALK_TOWARD  = 8
WALK_AWAY    = 9
N_AI_ACTIONS = 10

ACTION_NAMES = {
    IDLE: "Idle", PUNCH: "Punch", KICK: "Kick", BLOCK: "Block",
    JUMP: "Jump", CROUCH: "Crouch", SPECIAL_A: "Fireball", SPECIAL_B: "Uppercut"
}

DAMAGE = {
    IDLE: 0, PUNCH: 8, KICK: 12, BLOCK: 0,
    JUMP: 0, CROUCH: 0, SPECIAL_A: 18, SPECIAL_B: 22
}

# Active hit window: action_timer must be in [lo, hi] for hit to register
HIT_WINDOW = {
    PUNCH:     (8,  13),
    KICK:      (10, 18),
    SPECIAL_A: (10, 23),
    SPECIAL_B: (12, 28),
}

HIT_RANGE   = 100
BLOCK_RANGE = 100

BG_TOP    = (20,  10,  40)
BG_BOT    = (40,  20,  70)
P1_COL    = (70,  130, 220)
P2_COL    = (220, 70,  70)
HEALTH_BG = (60,  20,  20)
HEALTH_FG = (80,  200, 80)
HEALTH_LO = (220, 80,  40)
WHITE     = (255, 255, 255)
YELLOW    = (255, 220, 0)
GRAY      = (120, 120, 120)
RED_FLASH = (220, 60,  60)


class Fighter:
    W, H = 50, 90

    def __init__(self, x, facing=1, color=P1_COL):
        self.start_x      = x
        self.x            = float(x)
        self.y            = float(GROUND_Y - self.H)
        self.vy           = 0.0
        self.health       = 100.0
        self.facing       = facing
        self.color        = color
        self.action       = IDLE
        self.action_timer = 0
        self.on_ground    = True
        self.hit_flash    = 0
        self.hit_landed   = False
        self._walk_intent = None

    def reset(self):
        self.x            = float(self.start_x)
        self.y            = float(GROUND_Y - self.H)
        self.vy           = 0.0
        self.health       = 100.0
        self.action       = IDLE
        self.action_timer = 0
        self.on_ground    = True
        self.hit_flash    = 0
        self.hit_landed   = False
        self._walk_intent = None

    def move(self, dx):
        self.x = max(float(ARENA_LEFT), min(float(ARENA_RIGHT), self.x + dx))

    def apply_action(self, action):
        if action in (WALK_TOWARD, WALK_AWAY):
            self._walk_intent = action
            return
        self._walk_intent = None
        # Don't interrupt attacks
        if self.action_timer > 0 and self.action in HIT_WINDOW:
            return
        self.action     = action
        self.hit_landed = False
        if action == PUNCH:
            self.action_timer = 15
        elif action == KICK:
            self.action_timer = 20
        elif action == JUMP and self.on_ground:
            self.vy           = JUMP_VEL
            self.on_ground    = False
            self.action_timer = 30
        elif action == CROUCH:
            self.action_timer = 8
        elif action == SPECIAL_A:
            self.action_timer = 25
        elif action == SPECIAL_B:
            self.action_timer = 30
            if self.on_ground:
                self.vy = JUMP_VEL * 0.7
        elif action == BLOCK:
            self.action_timer = 12
        else:
            self.action       = IDLE
            self.action_timer = 0

    def update(self, other):
        # Walk
        if self._walk_intent == WALK_TOWARD and self.action not in HIT_WINDOW:
            self.move(WALK_SPEED * self.facing)
        elif self._walk_intent == WALK_AWAY and self.action not in HIT_WINDOW:
            self.move(-WALK_SPEED * self.facing)
        self._walk_intent = None

        # Timer
        if self.action_timer > 0:
            self.action_timer -= 1
        else:
            self.action = IDLE

        # Gravity
        self.vy += GRAVITY
        self.y  += self.vy
        if self.y >= GROUND_Y - self.H:
            self.y         = float(GROUND_Y - self.H)
            self.vy        = 0.0
            self.on_ground = True
            if self.action == JUMP:
                self.action = IDLE

        # Face opponent
        self.facing = 1 if other.x > self.x else -1

        # Hit detection — window-based, one hit per swing
        damage = 0
        if self.action in HIT_WINDOW and not self.hit_landed:
            lo, hi = HIT_WINDOW[self.action]
            if lo <= self.action_timer <= hi:
                dist = abs(self.x - other.x)
                if dist < HIT_RANGE:
                    if other.action == BLOCK and dist < BLOCK_RANGE:
                        damage = DAMAGE[self.action] // 4
                    else:
                        damage = DAMAGE[self.action]
                    other.health    = max(0.0, other.health - damage)
                    other.hit_flash = 8
                    self.hit_landed = True

        if self.hit_flash > 0:
            self.hit_flash -= 1

        return damage

    def draw(self, surf):
        col = RED_FLASH if self.hit_flash > 0 else self.color
        pygame.draw.rect(surf, col,
                         (int(self.x), int(self.y), self.W, self.H),
                         border_radius=6)
        head_x = int(self.x + self.W // 2)
        head_y = int(self.y) - 20
        pygame.draw.circle(surf, col, (head_x, head_y), 16)
        eye_dx = 5 * self.facing
        pygame.draw.circle(surf, WHITE, (head_x + eye_dx, head_y - 3), 4)
        pygame.draw.circle(surf, (10, 10, 50),
                           (head_x + eye_dx + self.facing, head_y - 3), 2)
        if self.action in HIT_WINDOW:
            pts = self._attack_shape()
            if pts:
                pygame.draw.polygon(surf, YELLOW, pts)

    def _attack_shape(self):
        cx = int(self.x + self.W // 2 + self.facing * (self.W // 2 + 10))
        cy = int(self.y + self.H // 2)
        if self.action == PUNCH:
            return [(cx, cy-6), (cx + self.facing*28, cy), (cx, cy+6)]
        if self.action == KICK:
            return [(cx, cy+10), (cx + self.facing*34, cy+20), (cx, cy+30)]
        if self.action == SPECIAL_A:
            return [(cx + int(16*math.cos(math.radians(i*60))),
                     cy + int(16*math.sin(math.radians(i*60)))) for i in range(6)]
        if self.action == SPECIAL_B:
            return [(cx-10, cy+20), (cx + self.facing*22, cy-34), (cx+10, cy+20)]
        return None


class FighterEnv:
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, render_mode="human"):
        self.render_mode  = render_mode
        self._screen      = None
        self._clock       = None
        self._font        = None
        self._sfont       = None

        self.p1 = Fighter(180, facing=1,  color=P1_COL)
        self.p2 = Fighter(620, facing=-1, color=P2_COL)

        self.round_frame  = 0
        self.max_frames   = ROUND_TIME * FPS
        self.round_number = 1
        self.p1_wins      = 0
        self.p2_wins      = 0
        self.done         = False

        if render_mode == "human":
            self._init_display()

    def _init_display(self):
        pygame.init()
        self._screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Fighter AI — Concept A")
        self._clock  = pygame.time.Clock()
        self._font   = pygame.font.SysFont("monospace", 22, bold=True)
        self._sfont  = pygame.font.SysFont("monospace", 14)

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.round_frame = 0
        self.done        = False
        return self._get_obs(), {}

    def step(self, p2_action: int):
        assert not self.done, "Call reset() after episode ends."

        if self.render_mode == "human":
            p1_action = self._read_keyboard()
        else:
            p1_action = np.random.randint(N_ACTIONS)

        self.p1.apply_action(p1_action)
        self.p2.apply_action(p2_action)
        self.p1.update(self.p2)
        self.p2.update(self.p1)
        self.round_frame += 1

        reward    = 0.0
        truncated = self.round_frame >= self.max_frames

        if self.p1.health <= 0 or self.p2.health <= 0 or truncated:
            if self.p2.health > self.p1.health:
                self.p2_wins += 1
                reward = 1.0
            elif self.p1.health > self.p2.health:
                self.p1_wins += 1
                reward = -1.0
            self.done = True

        obs  = self._get_obs()
        info = {
            "p1_action":   p1_action,
            "p2_action":   p2_action,
            "round":       self.round_number,
            "p1_wins":     self.p1_wins,
            "p2_wins":     self.p2_wins,
            "round_frame": self.round_frame,
        }

        # Render to surface — caller does display.flip()
        if self.render_mode in ("human", "rgb_array"):
            self._render_frame()

        return obs, reward, self.done, truncated, info

    def _get_obs(self):
        dist         = abs(self.p1.x - self.p2.x)
        health_delta = self.p1.health - self.p2.health
        round_t      = self.round_frame / self.max_frames
        return {
            "p1_action":       self.p1.action,
            "p1_x":            self.p1.x,
            "p1_y":            self.p1.y,
            "p1_health":       self.p1.health,
            "p1_facing":       self.p1.facing,
            "p2_x":            self.p2.x,
            "p2_y":            self.p2.y,
            "p2_health":       self.p2.health,
            "p2_action":       self.p2.action,
            "distance":        dist,
            "health_delta":    health_delta,
            "round_time":      round_t,
            "p1_is_attacking": int(self.p1.action in HIT_WINDOW),
        }

    def _read_keyboard(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_j]:     return PUNCH
        if keys[pygame.K_k]:     return KICK
        if keys[pygame.K_l]:     return BLOCK
        if keys[pygame.K_SPACE]: return JUMP
        if keys[pygame.K_s]:     return CROUCH
        if keys[pygame.K_u]:     return SPECIAL_A
        if keys[pygame.K_i]:     return SPECIAL_B
        if keys[pygame.K_d]:     return WALK_TOWARD
        if keys[pygame.K_a]:     return WALK_AWAY
        return IDLE

    def _render_frame(self):
        if self._screen is None:
            self._init_display()
        surf = self._screen
        surf.fill(BG_TOP)
        pygame.draw.rect(surf, BG_BOT, (0, GROUND_Y, W, H - GROUND_Y))
        pygame.draw.line(surf, (80, 60, 120), (0, GROUND_Y), (W, GROUND_Y), 2)
        self._draw_health(surf, self.p1.health, 30,      20, left=True,  label="P1")
        self._draw_health(surf, self.p2.health, W - 230, 20, left=False, label="AI")
        secs   = max(0, ROUND_TIME - self.round_frame // FPS)
        t_surf = self._font.render(f"{secs:02d}", True, WHITE)
        surf.blit(t_surf, (W // 2 - t_surf.get_width() // 2, 14))
        info_str = f"Round {self.round_number}   P1: {self.p1_wins}   AI: {self.p2_wins}"
        i_surf = self._sfont.render(info_str, True, GRAY)
        surf.blit(i_surf, (W // 2 - i_surf.get_width() // 2, 44))
        self.p1.draw(surf)
        self.p2.draw(surf)
        ctrl = "A/D=Walk  J=Punch  K=Kick  L=Block  Space=Jump  S=Crouch  U=Fireball  I=Uppercut"
        c_surf = self._sfont.render(ctrl, True, GRAY)
        surf.blit(c_surf, (W // 2 - c_surf.get_width() // 2, H - 22))
        # NOTE: caller calls pygame.display.flip()

    def _draw_health(self, surf, hp, x, y, left=True, label="P1"):
        bar_w  = 200
        filled = int(bar_w * max(0.0, hp) / 100.0)
        col    = HEALTH_FG if hp > 30 else HEALTH_LO
        pygame.draw.rect(surf, HEALTH_BG, (x, y, bar_w, 20), border_radius=4)
        if filled > 0:
            if left:
                pygame.draw.rect(surf, col, (x, y, filled, 20), border_radius=4)
            else:
                pygame.draw.rect(surf, col,
                                 (x + bar_w - filled, y, filled, 20),
                                 border_radius=4)
        txt   = f"{label} {int(hp)}"
        lsurf = self._sfont.render(txt, True, WHITE)
        surf.blit(lsurf, (x + bar_w // 2 - lsurf.get_width() // 2, y + 2))

    def get_rgb_frame(self):
        if self._screen is None:
            return None
        return pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)

    def close(self):
        if self._screen:
            pygame.quit()
            self._screen = None