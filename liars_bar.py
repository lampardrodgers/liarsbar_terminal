#!/usr/bin/env python3
"""
骗子酒馆（终端版）：
- 1 名人类玩家 + 3 名 Bot
- 卡组：6×Q, 6×K, 6×A, 2×Joker
- 每轮开始会确定一个“真牌”，玩家默认宣称自己出的牌就是它
- 任意人质疑后立即检验、触发俄轮，并重新洗牌/换真牌，由被质疑者下家开新局
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

RANKS: Sequence[str] = ("Q", "K", "A")
DISPLAY_RANKS: Sequence[str] = ("Q", "K", "A", "Joker")
BOT_NAMES: Sequence[str] = ("Bot-GPT", "Bot-GEMINI", "Bot-DEEPSEEK")
HAND_SIZE = 5
DEFAULT_DELAY = 1.0

RANK_ORDER = {"Q": 0, "K": 1, "A": 2, "Joker": 3}
TOTAL_RANK_COUNTS = {"Q": 6, "K": 6, "A": 6, "Joker": 2}
COLOR_RESET = "\033[0m"
RANK_COLORS = {
    "Q": "\033[93m",  # yellow
    "K": "\033[94m",  # blue
    "A": "\033[95m",  # magenta
    "Joker": "\033[91m",  # red
}
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
USE_COLOR = sys.stdout.isatty()
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def format_rank(rank: str) -> str:
    label = "Joker" if rank == "Joker" else rank
    if USE_COLOR and rank in RANK_COLORS:
        return f"{RANK_COLORS[rank]}{label}{COLOR_RESET}"
    return label


def colorize(text: str, color: str) -> str:
    if USE_COLOR:
        return f"{color}{text}{COLOR_RESET}"
    return text


def style(text: str, color: Optional[str] = None, bold: bool = False) -> str:
    if not USE_COLOR and not bold:
        return text
    prefix = ""
    if color:
        prefix += color
    if bold:
        prefix += "\033[1m"
    return f"{prefix}{text}{COLOR_RESET}"


def format_card_names(cards: Sequence["Card"]) -> str:
    if not cards:
        return "无"
    return " ".join(format_rank(card.rank) for card in cards)


def sort_cards(cards: List["Card"]) -> None:
    cards.sort(key=lambda c: (RANK_ORDER.get(c.rank, 99), c.uid))


@dataclass(frozen=True)
class Card:
    rank: str
    uid: int

    @property
    def symbol(self) -> str:
        return "Joker" if self.rank == "Joker" else self.rank


def create_deck() -> List[Card]:
    deck: List[Card] = []
    uid = 1
    for rank in RANKS:
        for _ in range(6):
            deck.append(Card(rank=rank, uid=uid))
            uid += 1
    for _ in range(2):
        deck.append(Card(rank="Joker", uid=uid))
        uid += 1
    random.shuffle(deck)
    return deck


class Player:
    def __init__(self, name: str, is_bot: bool):
        self.name = name
        self.is_bot = is_bot
        self.hand: List[Card] = []
        self.alive = True
        self.shots_taken = 0
        self.bullet_slot = random.randint(1, 6)  # 第几次扣扳机必出弹

    def card_display(self) -> str:
        if not self.hand:
            return "(空)"
        sort_cards(self.hand)
        labels = []
        for idx, card in enumerate(self.hand, start=1):
            labels.append(f"<{idx}>{format_rank(card.rank)}")
        return " ".join(labels)

    def remove_cards(self, cards: List[Card]) -> None:
        for card in cards:
            self.hand.remove(card)


class HumanPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name or "玩家", is_bot=False)


class BotPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name, is_bot=True)
        self.truth_bias = random.uniform(0.55, 0.8)
        self.risk_tolerance = random.uniform(0.45, 0.75)
        self.bluff_spice = random.uniform(0.2, 0.45)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def _is_truthful(cards: Sequence[Card], declared_rank: str) -> bool:
        return all(card.rank == declared_rank or card.rank == "Joker" for card in cards)

    def _should_allin_truthful(self, game: "LiarsBarGame", target_rank: str) -> bool:
        """1v1 且手牌都能诚实宣称 target_rank 时，决定是否一口气打光。"""
        # 只有一张时没得选
        if len(self.hand) <= 1:
            return True
        opponent = next((p for p in game.players if p is not self and len(p.hand) > 0), None)
        if opponent is None:
            return False
        pressure = self._phase_pressure(game)
        aggro = self.risk_tolerance + self.bluff_spice * 0.15 + pressure * 0.25
        # 如果自己牌更少，或压力大，或侵略性高，则更愿意 all-in
        behind_or_even = len(self.hand) <= len(opponent.hand)
        if pressure > 0.7 or aggro > 0.7:
            return True
        if behind_or_even and pressure > 0.5:
            return True
        # 否则留点余地，30% 概率尝试一把
        return random.random() < 0.3

    def choose_play(self, game: "LiarsBarGame") -> tuple[List[Card], str]:
        if not self.hand:
            return [], game.true_rank or random.choice(RANKS)
        max_cards = min(3, len(self.hand))

        declared_rank = game.true_rank or random.choice(RANKS)
        # 终局 1v1（仅剩两名有牌玩家）且手牌全可诚实时，视侵略性选择全下或慢打。
        active_with_cards = [p for p in game.players if len(p.hand) > 0]
        if len(active_with_cards) == 2 and self in active_with_cards and len(self.hand) <= max_cards:
            target_rank = game.true_rank or next((c.rank for c in self.hand if c.rank != "Joker"), declared_rank)
            if self._is_truthful(self.hand, target_rank):
                if self._should_allin_truthful(game, target_rank):
                    return self.hand.copy(), target_rank
                # 慢打：保持 declared_rank 与 target_rank 一致以继续诚实出
                declared_rank = target_rank

        matching = [card for card in self.hand if card.rank == declared_rank]
        jokers = [card for card in self.hand if card.rank == "Joker"]
        non_target = [card for card in self.hand if card.rank not in (declared_rank, "Joker")]

        phase_pressure = self._phase_pressure(game)
        truthful_chance = self.truth_bias + phase_pressure * 0.12
        truthful_chance -= (self.bluff_spice - 0.25) * 0.35
        if len(matching) >= 2:
            truthful_chance += 0.1
        remaining_for_target = game.estimate_remaining(declared_rank, exclude=len(matching))
        if remaining_for_target <= 1:
            truthful_chance += 0.15
        elif remaining_for_target >= 4 and len(non_target) > len(matching):
            truthful_chance -= self.bluff_spice * 0.15
        if not matching and not jokers:
            truthful_chance = 0.0
        if self.shots_taken >= 2:
            truthful_chance += 0.05
        truthful_chance = min(max(truthful_chance, 0.0), 0.95)
        bluff_possible = bool(non_target) or (len(jokers) > 0 and len(self.hand) > len(jokers))
        truthful = True if not bluff_possible else random.random() < truthful_chance

        if truthful and (matching or jokers):
            cards = self._pick_truthful_cards(matching, jokers, max_cards, game)
        else:
            cards = self._pick_bluff_cards(non_target, max_cards, game)

        # 避免“出完就被迫质疑”时的自杀式 bluff：如果这一手会把牌出光且别人还有牌，就尽量保证诚实或至少留一张。
        others_have_cards = any(len(p.hand) > 0 for p in game.players if p is not self)
        will_empty = len(cards) == len(self.hand)
        if will_empty and others_have_cards and not self._is_truthful(cards, declared_rank):
            if matching or jokers:
                cards = self._pick_truthful_cards(matching, jokers, max_cards, game)
                declared_rank = game.true_rank or declared_rank
            elif len(self.hand) > 1:
                safe_max = min(3, len(self.hand) - 1)
                cards = self._pick_bluff_cards(non_target, safe_max, game)
        return cards, declared_rank

    def _pick_truthful_cards(
        self, matching: List[Card], jokers: List[Card], max_cards: int, game: "LiarsBarGame"
    ) -> List[Card]:
        pool_size = len(matching) + len(jokers)
        limit = min(max_cards, pool_size)
        count = random.randint(1, max(1, limit))
        cards: List[Card] = []
        if matching:
            take_truth = min(len(matching), count)
            cards.extend(random.sample(matching, take_truth))
        needed = count - len(cards)
        if needed <= 0:
            return cards
        shuffled_jokers = jokers[:]
        random.shuffle(shuffled_jokers)
        reserve = min(self._joker_reserve_size(game, len(shuffled_jokers)), len(shuffled_jokers))
        usable = len(shuffled_jokers) - reserve
        if len(cards) + usable < count:
            usable = len(shuffled_jokers)
        take = min(usable, needed)
        cards.extend(shuffled_jokers[:take])
        needed -= take
        if needed > 0:
            cards.extend(shuffled_jokers[take:take + needed])
        return cards

    def _pick_bluff_cards(self, non_target: List[Card], max_cards: int, game: "LiarsBarGame") -> List[Card]:
        pressure = self._phase_pressure(game)
        upper = max(1, max_cards - (1 if pressure > 0.6 else 0))
        count = random.randint(1, upper)
        cards: List[Card] = []
        if len(non_target) >= count:
            cards = random.sample(non_target, count)
        else:
            cards = non_target.copy()
            needed = count - len(cards)
            filler_pool = [card for card in self.hand if card not in cards and card.rank != "Joker"]
            random.shuffle(filler_pool)
            take = min(needed, len(filler_pool))
            cards.extend(filler_pool[:take])
            needed -= take
            if needed > 0:
                leftovers = [card for card in self.hand if card not in cards and card.rank != "Joker"]
                cards.extend(leftovers[:needed])
        return cards

    def _phase_pressure(self, game: "LiarsBarGame") -> float:
        alive_ratio = len(game.players) / max(len(BOT_NAMES) + 1, 1)
        hand_pressure = (HAND_SIZE - len(self.hand)) / HAND_SIZE
        trigger_pressure = min(self.shots_taken, 3) / 3
        pressure = 0.4 * (1 - alive_ratio) + 0.35 * hand_pressure + 0.25 * trigger_pressure
        return max(0.0, min(pressure, 1.0))

    def _joker_reserve_size(self, game: "LiarsBarGame", joker_qty: int) -> int:
        if joker_qty <= 1:
            return 0
        pressure = self._phase_pressure(game)
        if pressure >= 0.7:
            return min(2, joker_qty - 1)
        if pressure >= 0.4:
            return 1
        return 1 if joker_qty > 2 else 0

    def want_to_call(self, game: "LiarsBarGame", claim: "Claim") -> bool:
        if claim.player == self:
            return False
        target_rank = claim.declared_rank
        matching = sum(1 for card in self.hand if card.rank == target_rank)
        joker_count = sum(1 for card in self.hand if card.rank == "Joker")
        own_support = matching + joker_count
        deficit = max(claim.declared_qty - own_support, 0)

        others_target = game.estimate_remaining(target_rank, exclude=matching)
        others_jokers = game.estimate_remaining("Joker", exclude=joker_count)
        max_possible = others_target + others_jokers
        if claim.declared_qty > max_possible:
            return True

        suspicion = 0.0
        suspicion += 0.3 * (deficit / max(claim.declared_qty, 1))
        profile = game.get_profile(claim.player)
        suspicion += (0.55 - profile.honesty) * 0.5
        suspicion += 0.05 * (claim.declared_qty - 1)
        suspicion += (self.bluff_spice - 0.3) * 0.2
        if game.true_rank and target_rank != game.true_rank:
            suspicion += 0.04
        revealed_ratio = 0.0
        total_rank_cards = TOTAL_RANK_COUNTS.get(target_rank, 1)
        revealed_ratio = game.revealed_counts.get(target_rank, 0) / total_rank_cards
        suspicion += revealed_ratio * 0.2

        my_profile = game.get_profile(self)
        missed_calls = my_profile.calls_made - my_profile.successful_calls
        suspicion -= missed_calls * 0.03

        pressure = self._phase_pressure(game)
        caution = min(self.shots_taken, 3) * (0.15 * (1 - self.risk_tolerance))
        suspicion -= caution
        suspicion += min(claim.player.shots_taken, 3) * 0.05
        suspicion += pressure * 0.08

        if len(self.hand) <= 2:
            suspicion += 0.05
        elif len(self.hand) >= 4:
            suspicion -= 0.05

        call_prob = min(max(self._sigmoid(suspicion), 0.05), 0.95)
        return random.random() < call_prob

    def __repr__(self) -> str:
        return f"BotPlayer({self.name})"


@dataclass
class PlayerProfile:
    truthful_claims: int = 0
    lies: int = 0
    calls_made: int = 0
    successful_calls: int = 0

    @property
    def honesty(self) -> float:
        total = self.truthful_claims + self.lies
        if total == 0:
            return 0.5
        return self.truthful_claims / total

    @property
    def call_success(self) -> float:
        if self.calls_made == 0:
            return 0.5
        return self.successful_calls / self.calls_made


@dataclass
class Claim:
    player: Player
    declared_rank: str
    cards: List[Card]

    @property
    def declared_qty(self) -> int:
        return len(self.cards)

    def truthful(self) -> bool:
        return all(card.rank == self.declared_rank or card.rank == "Joker" for card in self.cards)

    def reveal_text(self) -> str:
        return f"{self.player.name} 实际出牌: {format_card_names(self.cards)}"


class LiarsBarGame:
    def __init__(self, human_name: str, delay: float):
        self.delay = delay
        self.players: List[Player] = []
        self.current_idx = 0
        self.true_rank: Optional[str] = None
        self.last_claim: Optional[Claim] = None
        self.last_claim_position: Optional[int] = None
        self.round_no = 1
        self.human_name = human_name or "玩家"
        self.player_profiles: Dict[Player, PlayerProfile] = {}
        self.revealed_counts: Dict[str, int] = {rank: 0 for rank in DISPLAY_RANKS}
        self.show_bot_hands = False
        self.log_path = Path(__file__).resolve().parent / "log.txt"
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.log(f"=== 新的游戏开始 === {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log(self, message: str) -> None:
        if not getattr(self, "log_file", None) or self.log_file.closed:
            return
        lines = message.splitlines() or [""]
        for line in lines:
            self.log_file.write(strip_ansi(line) + "\n")
        self.log_file.flush()

    def close_log(self) -> None:
        if getattr(self, "log_file", None) and not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()

    def slow_print(self, message: str = "", with_delay: bool = True) -> None:
        print(message)
        self.log(message)
        if with_delay and self.delay > 0:
            time.sleep(self.delay)

    def get_profile(self, player: Player) -> PlayerProfile:
        profile = self.player_profiles.get(player)
        if profile is None:
            profile = PlayerProfile()
            self.player_profiles[player] = profile
        return profile

    def reset_revealed_counts(self) -> None:
        self.revealed_counts = {rank: 0 for rank in DISPLAY_RANKS}

    def track_revealed_cards(self, cards: Sequence[Card]) -> None:
        for card in cards:
            self.revealed_counts[card.symbol] = self.revealed_counts.get(card.symbol, 0) + 1

    def show_revealed_summary(self) -> None:
        lines = ["已出牌统计："]
        for rank in DISPLAY_RANKS:
            count = self.revealed_counts.get(rank, 0)
            lines.append(f"  {format_rank(rank)}: {count}")
        self.slow_print("\n".join(lines), with_delay=False)

    def show_all_hands_and_revealed(self) -> None:
        self.slow_print("=== 全部公开信息 ===", with_delay=False)
        for p in self.players:
            self.slow_print(f"{p.name} 手牌：{p.card_display()}", with_delay=False)
        self.show_revealed_summary()

    def log_hands_state(self, reason: str) -> None:
        lines = [f"[隐藏信息] {reason}"]
        for p in self.players:
            hand_copy = p.hand[:]
            sort_cards(hand_copy)
            lines.append(f"{p.name} 手牌({len(hand_copy)}): {format_card_names(hand_copy)}")
        revealed = " ".join(f"{rank}:{self.revealed_counts.get(rank, 0)}" for rank in DISPLAY_RANKS)
        lines.append(f"已出牌统计：{revealed}")
        self.log("\n".join(lines))

    def estimate_remaining(self, rank: str, exclude: int = 0) -> int:
        total = TOTAL_RANK_COUNTS.get(rank, 0)
        revealed = self.revealed_counts.get(rank, 0)
        return max(total - revealed - exclude, 0)

    def should_force_call(self, player: Player) -> bool:
        if not self.last_claim or not player.hand:
            return False
        active_with_cards = [p for p in self.players if len(p.hand) > 0]
        # 仅剩当前玩家握有牌，则必须质疑；否则正常流程
        return len(active_with_cards) == 1 and active_with_cards[0] is player

    def record_claim_outcome(self, claim: Claim, truthful: bool, caller: Player) -> None:
        claimer_profile = self.get_profile(claim.player)
        if truthful:
            claimer_profile.truthful_claims += 1
        else:
            claimer_profile.lies += 1
        caller_profile = self.get_profile(caller)
        caller_profile.calls_made += 1
        if not truthful:
            caller_profile.successful_calls += 1

    def setup(self) -> None:
        human = HumanPlayer(self.human_name)
        bots = [BotPlayer(name) for name in BOT_NAMES]
        self.players = [human] + bots
        self.player_profiles = {player: PlayerProfile() for player in self.players}
        self.slow_print("=== 骗子酒馆开局 ===", with_delay=False)
        self.slow_print("牌堆：6×Q, 6×K, 6×A, 2×Joker（共 20 张）", with_delay=False)
        self.slow_print(
            "操作提示：y=质疑上家；输入索引组合直接出牌（如 13 表示第1+3张）；show=切换Bot显牌；h=帮助。",
            with_delay=False,
        )
        self.start_new_round(0)

    def start_new_round(self, starter_idx: int) -> None:
        if len(self.players) <= 1:
            return
        deck = create_deck()
        for player in self.players:
            player.hand.clear()
        for _ in range(HAND_SIZE):
            for player in self.players:
                if not deck:
                    break
                player.hand.append(deck.pop())
        for player in self.players:
            sort_cards(player.hand)
        self.true_rank = random.choice(RANKS)
        self.reset_revealed_counts()
        self.last_claim = None
        self.last_claim_position = None
        self.current_idx = starter_idx % len(self.players)
        self.slow_print("\n=== 新的一轮开始 ===")
        self.slow_print(f"本轮真牌：{format_rank(self.true_rank)}")
        self.log_hands_state("发牌完成后手牌")

    def run(self) -> None:
        self.setup()
        while len(self.players) > 1:
            if self.current_idx >= len(self.players):
                self.current_idx = 0
            player = self.players[self.current_idx]
            next_idx = self.take_turn(player)
            if len(self.players) <= 1:
                break
            self.current_idx = next_idx % len(self.players)
        if self.players:
            winner = self.players[0]
            self.slow_print(f"游戏结束，{winner.name} 生还！", with_delay=False)
            # 展示获胜玩家左轮中的子弹位置，x/6 表示第 x 扣扳机会击发
            self.slow_print(f"子弹所在位置：{winner.bullet_slot}/6", with_delay=False)
        else:
            self.slow_print("所有玩家阵亡，没人赢。", with_delay=False)
        self.log("=== 本局结束 ===")
        self.close_log()

    def take_turn(self, player: Player) -> int:
        divider = "-" * 48
        self.slow_print(divider)
        self.slow_print(f"第 {self.round_no} 轮：{player.name} 的回合")
        self.show_public_state(current=player)
        if self.should_force_call(player):
            self.slow_print(f"{player.name} 是唯一仍握有手牌的玩家，被迫质疑。")
            idx = self.resolve_call(player)
            self.round_no += 1
            return idx
        if not player.hand:
            if self.last_claim:
                self.slow_print(f"{player.name} 没牌，只能质疑。")
                idx = self.resolve_call(player)
            else:
                self.slow_print(f"{player.name} 没牌，等待下家。")
                idx = self.advance_turn(start_from=player)
            self.round_no += 1
            return idx
        if player.is_bot:
            idx = self.bot_turn(player)
        else:
            idx = self.human_turn(player)
        self.round_no += 1
        return idx

    def show_public_state(self, current: Player) -> None:
        if self.true_rank:
            self.slow_print(f"本轮真牌：{format_rank(self.true_rank)}")
        if self.last_claim:
            declared = format_rank(self.last_claim.declared_rank)
            self.slow_print(
                f"桌面声明：{self.last_claim.player.name} 压了 {self.last_claim.declared_qty} 张（宣称 {declared}）",
                with_delay=False,
            )
        else:
            self.slow_print("桌面暂无声明。", with_delay=False)
        for player in self.players:
            if player.is_bot:
                if self.show_bot_hands:
                    hand_text = player.card_display()
                else:
                    hand_text = f"(隐藏，余 {len(player.hand)} 张)"
                self.slow_print(f"{player.name} 手牌：{hand_text}", with_delay=False)
        if not current.is_bot:
            self.slow_print(f"你的手牌：{current.card_display()}", with_delay=False)
        status_parts = []
        for p in self.players:
            cards_text = colorize(f"余牌:{len(p.hand)}", GREEN)
            trigger_text = colorize(f"扣扳机:{p.shots_taken}次", RED)
            status_parts.append(f"{p.name} {cards_text} {trigger_text}")
        status_line = " | ".join(status_parts)
        self.slow_print(status_line, with_delay=False)
        self.log_hands_state(f"回合 {self.round_no}，当前玩家 {current.name} 前的实际手牌")
        if self.show_bot_hands:
            self.show_revealed_summary()

    def bot_turn(self, bot: BotPlayer) -> int:
        if self.last_claim and self.last_claim.player != bot:
            if bot.want_to_call(self, self.last_claim):
                shout = style(f"{bot.name} 大喊：骗子！", PURPLE, bold=True)
                self.slow_print(shout)
                return self.resolve_call(bot)
            trust = style(f"{bot.name} 暂且相信，继续出牌。", BLUE, bold=True)
            self.slow_print(trust)
        elif self.last_claim and self.last_claim.player == bot:
            self.slow_print(f"{bot.name} 轮到自己的声明，继续加注。")
        cards, declared_rank = bot.choose_play(self)
        if not cards:
            if self.last_claim:
                self.slow_print(f"{bot.name} 无牌可出，被迫质疑。")
                return self.resolve_call(bot)
            self.slow_print(f"{bot.name} 无牌，只能等待。")
            return self.advance_turn(start_from=bot)
        bot.remove_cards(cards)
        claim = Claim(player=bot, declared_rank=declared_rank, cards=cards)
        self.announce_play(claim)
        self.set_last_claim(claim)
        return self.advance_turn(start_from=bot)

    def human_turn(self, player: HumanPlayer) -> int:
        while True:
            choice = input("你的动作（y=质疑，数字=出牌，show=显/隐Bot，h=帮助）：").strip().lower()
            if choice == "h":
                self.slow_print(
                    "示例：<1>Joker <2>Q <3>Q；输入23表示出第2和第3张，最多 3 张。输入 show 切换是否展示 Bot 手牌。",
                    with_delay=False,
                )
                continue
            if choice == "show":
                self.show_bot_hands = not self.show_bot_hands
                state = "开启" if self.show_bot_hands else "关闭"
                self.slow_print(f"Bot 手牌展示已{state}。", with_delay=False)
                if self.show_bot_hands:
                    self.show_all_hands_and_revealed()
                continue
            if choice == "y":
                if not self.last_claim:
                    self.slow_print("当前没有可以质疑的声明。", with_delay=False)
                    continue
                return self.resolve_call(player)
            if not choice:
                self.slow_print("请输入动作。", with_delay=False)
                continue
            indexes = self.parse_card_indexes(choice, len(player.hand))
            if not indexes:
                self.slow_print("索引无效或超出范围。", with_delay=False)
                continue
            if len(indexes) > 3:
                self.slow_print("一次最多出 3 张。", with_delay=False)
                continue
            cards = self.extract_cards_by_indexes(player, indexes)
            claim = Claim(player=player, declared_rank=self.true_rank or "Q", cards=cards)
            self.announce_play(claim)
            self.set_last_claim(claim)
            return self.advance_turn(start_from=player)

    def parse_card_indexes(self, raw: str, hand_size: int) -> List[int]:
        stripped = raw.replace(",", " ").replace("-", " ").replace(".", " ").split()
        indexes: List[int] = []
        if not stripped and raw.isdigit():
            indexes = [int(ch) for ch in raw]
        elif raw.isdigit():
            indexes = [int(ch) for ch in raw]
        else:
            for token in stripped or [raw]:
                if token.isdigit():
                    indexes.append(int(token))
        indexes = [idx for idx in indexes if 1 <= idx <= hand_size]
        seen = set()
        ordered: List[int] = []
        for idx in indexes:
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)
        return ordered

    def extract_cards_by_indexes(self, player: Player, indexes: List[int]) -> List[Card]:
        cards: List[Card] = []
        for idx in sorted(indexes, reverse=True):
            cards.append(player.hand.pop(idx - 1))
        cards.reverse()
        return cards

    def announce_play(self, claim: Claim) -> None:
        card_text = format_card_names(claim.cards)
        self.slow_print(
            f"{claim.player.name} 压下 {claim.declared_qty} 张，目标真牌 {format_rank(claim.declared_rank)}"
        )
        if claim.player.is_bot and not self.show_bot_hands:
            hidden_note = f"(隐藏，余{len(claim.player.hand)}张；输入 show 查看)"
            self.slow_print(f"实际牌面：{hidden_note}", with_delay=False)
        else:
            self.slow_print(f"实际牌面：{card_text}", with_delay=False)
        self.log(
            f"[实际出牌] {claim.player.name} 声称 {claim.declared_rank}，实际牌：{card_text}（余牌 {len(claim.player.hand)}）"
        )
        self.track_revealed_cards(claim.cards)
        self.log_hands_state("出牌后手牌状态")

    def set_last_claim(self, claim: Claim) -> None:
        self.last_claim = claim
        self.last_claim_position = self.players.index(claim.player)

    def resolve_call(self, caller: Player) -> int:
        assert self.last_claim is not None
        claim = self.last_claim
        snapshot_idx = (
            self.last_claim_position
            if self.last_claim_position is not None
            else self.players.index(claim.player)
        )
        call_line = style(f"{caller.name} 质疑 {claim.player.name}！翻看牌面……", PURPLE, bold=True)
        self.slow_print(call_line)
        self.slow_print(f"真牌是：{format_rank(self.true_rank or 'Q')}")
        self.slow_print(claim.reveal_text())
        truthful = claim.truthful()
        victim = caller if truthful else claim.player
        self.slow_print(f"判定：{claim.player.name} {'诚实' if truthful else '撒谎'}。")
        self.record_claim_outcome(claim, truthful, caller)
        if truthful:
            self.slow_print(f"{caller.name} 必须扣动扳机。")
        else:
            self.slow_print(f"{claim.player.name} 必须扣动扳机。")
        died = self.pull_trigger(victim)
        self.last_claim = None
        self.last_claim_position = None
        if len(self.players) <= 1:
            return 0
        starter_idx = self.compute_next_starter(claim.player, snapshot_idx)
        self.start_new_round(starter_idx)
        return self.current_idx

    def compute_next_starter(self, claimant: Player, snapshot_idx: int) -> int:
        if not self.players:
            return 0
        if claimant in self.players:
            idx = self.players.index(claimant)
            return (idx + 1) % len(self.players)
        if self.players:
            return snapshot_idx % len(self.players)
        return 0

    def pull_trigger(self, target: Player) -> bool:
        self.slow_print("旋转左轮 ...")
        target.shots_taken += 1
        if target.shots_taken >= target.bullet_slot:
            self.slow_print(f"Bang! {target.name} 中弹身亡！")
            self.eliminate_player(target)
            return True
        self.slow_print(f"Click! {target.name} 逃过一劫。（{target.shots_taken}/6）")
        return False

    def eliminate_player(self, target: Player) -> None:
        if target not in self.players:
            return
        idx = self.players.index(target)
        self.players.pop(idx)
        target.alive = False
        self.slow_print(f"{target.name} 出局，剩余 {len(self.players)} 人。")
        target.shots_taken = 0
        target.bullet_slot = random.randint(1, 6)
        if idx < self.current_idx:
            self.current_idx -= 1
        elif idx == self.current_idx:
            self.current_idx %= len(self.players) if self.players else 0

    def advance_turn(self, start_from: Optional[Player] = None) -> int:
        if not self.players:
            return 0
        if start_from is None:
            start_idx = self.current_idx
        else:
            if start_from in self.players:
                start_idx = self.players.index(start_from)
            else:
                start_idx = self.current_idx
        return (start_idx + 1) % len(self.players)


def main() -> None:
    parser = argparse.ArgumentParser(description="骗子酒馆终端版")
    parser.add_argument("--name", help="玩家称呼", default="玩家")
    parser.add_argument("--fast", action="store_true", help="加快节奏（延时 0.2 秒）")
    args = parser.parse_args()
    delay = 0.2 if args.fast else DEFAULT_DELAY
    game = LiarsBarGame(human_name=args.name, delay=delay)
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n手动退出。")
        sys.exit(0)
    finally:
        game.close_log()


if __name__ == "__main__":
    main()
