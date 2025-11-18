# liarsbar_terminal

这是一个终端版的《骗子酒馆》桌游模拟器，包含一名人类玩家与三名 Bot，使用 6×Q/K/A 与 2 张 Joker 构筑的牌组，支持质疑与俄轮等核心机制。
This is a terminal adaptation of the "Liar's Bar" board game, featuring one human player and three Bots, built with a deck of 6×Q/K/A plus two Jokers, and it supports core mechanics such as challenges and Russian roulette.

## 运行

1. 确保使用 Python 3.11+（依赖 `dataclasses` 与 `typing` 标准库）；
   Make sure you are running Python 3.11+ (the game depends on the standard libraries `dataclasses` and `typing`).
2. 在项目根目录运行：
   Run the following command from the project root:

```bash
python3 liars_bar.py [--name 玩家名] [--fast]
```

参数说明：
- `--name`：指定玩家称呼，默认 `玩家`；
  `--name`: Sets the player name (defaults to "Player").
- `--fast`：加速模式，将 UI 延时调为 0.2 秒。
  `--fast`: Enables fast mode by shortening UI delays to 0.2 seconds.

## 文件

- `liars_bar.py`：游戏主程序，包含玩家交互、Bot 策略与日志记录；
  `liars_bar.py`: The game entry point that handles player interaction, Bot decision logic, and log writing.
- `log.txt`：一次游戏的记录示例，可直接删除或重新生成；
  `log.txt`: A sample log from a single playthrough; safe to delete or regenerate.
- `__pycache__/`：运行时缓存，已在 `.gitignore` 中排除。
  `__pycache__/`: Runtime cache already excluded via `.gitignore`.

## 目标

创建一个尽可能贴近桌游规则的命令行体验，欢迎直接在终端调整参数或参考日志排查行为。
Build a command-line experience that closely mirrors the tabletop rules; feel free to tweak options or inspect the logs directly in the terminal.
