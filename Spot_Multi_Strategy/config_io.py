import json
from dataclasses import asdict, fields
from pathlib import Path

from config import StrategyConfig


def save_strategy_config(strategy_config, file_path):
    # 把策略参数保存为可以重复使用的JSON文件
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            asdict(strategy_config),
            file,
            ensure_ascii=False,
            indent=2
        )


def load_strategy_config(file_path):
    # 从JSON文件恢复策略参数
    input_path = Path(file_path)

    with input_path.open("r", encoding="utf-8") as file:
        parameter_data = json.load(file)

    if not isinstance(parameter_data, dict):
        raise ValueError("策略参数文件必须是JSON对象")

    allowed_names = {
        field.name
        for field in fields(StrategyConfig)
    }
    unknown_names = set(parameter_data) - allowed_names

    if unknown_names:
        raise ValueError(
            f"策略参数文件存在未知字段：{sorted(unknown_names)}"
        )

    return StrategyConfig(**parameter_data)
