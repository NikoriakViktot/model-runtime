def build_context(snippets: list[dict]) -> str:
    lines = []
    for s in snippets:
        topic = s.get("topic", "general")
        text = s.get("text", "")
        lines.append(f"[{topic}] {text}")
    return "\n".join(lines)
