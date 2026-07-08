import json
from datetime import datetime, timezone
from pathlib import Path


class MemoryManager:
    def __init__(self, memory_file_name: str = "./memory.json"):
        self.memory_file_path = (Path(__file__).resolve().parent / memory_file_name).resolve()

    def load_memories(self) -> dict:
        try:
            data = json.loads(self.memory_file_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"memories": [], "conversationHistory": []}

        if not isinstance(data.get("memories"), list):
            data["memories"] = []
        if not isinstance(data.get("conversationHistory"), list):
            data["conversationHistory"] = []

        return data

    def save_memories(self, data: dict) -> None:
        self.memory_file_path.write_text(json.dumps(data, indent=2))

    def add_memory(self, type: str, key: str, value: str, source: str = "user") -> None:
        data = self.load_memories()

        norm_type = type.strip().lower()
        norm_key = key.strip().lower()
        norm_value = value.strip()

        existing = next(
            (m for m in data["memories"] if m["type"] == norm_type and m["key"].lower() == norm_key),
            None,
        )

        if existing is not None:
            if existing["value"] != norm_value:
                existing["value"] = norm_value
                existing["timestamp"] = datetime.now(timezone.utc).isoformat()
                existing["source"] = source
                print(f"Updated memory: {norm_key} -> {norm_value}")
            else:
                print(f"Skipped duplicate memory: {norm_key}")
        else:
            data["memories"].append(
                {
                    "type": norm_type,
                    "key": norm_key,
                    "value": norm_value,
                    "source": source,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"Added memory: {norm_key} = {norm_value}")

        self.save_memories(data)

    def get_memory_summary(self) -> str:
        data = self.load_memories()
        facts = [m for m in data["memories"] if m["type"] == "fact"]
        prefs = [m for m in data["memories"] if m["type"] == "preference"]

        summary = "\n=== LONG-TERM MEMORY ===\n"

        if facts:
            summary += "\nKnown Facts:\n"
            for f in facts:
                summary += f"- {f['key']}: {f['value']}\n"

        if prefs:
            summary += "\nUser Preferences:\n"
            for p in prefs:
                summary += f"- {p['key']}: {p['value']}\n"

        return summary
