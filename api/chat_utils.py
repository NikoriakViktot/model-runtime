def inject_context(
    epitaph: str,
    context: str,
    chat_history: list[dict]
) -> list[dict]:

    system_msg = {
        "role": "system",
        "content": f"""
You are {epitaph}.
The following is relevant past context from your memory graph.
Use it to answer consistently and accurately.

Context:
{context}
"""
    }

    return [system_msg] + chat_history
