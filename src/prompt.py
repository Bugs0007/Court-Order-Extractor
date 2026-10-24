"""One prompt, shared by Gemini labelling, training, and evaluation, so the model
is always asked the same question in the same words."""
from .schema import OUTCOMES, PARTIES

MAX_CHARS = 6000  # ~1,500-2,000 Llama tokens; keeps prompt + answer under max_seq_length

SYSTEM = (
    "You extract structured data from Indian court orders. "
    "Reply with one JSON object and nothing else, with exactly these keys: "
    "case_number (string or null), order_date (YYYY-MM-DD or null), "
    "next_hearing_date (YYYY-MM-DD or null), "
    f"outcome (one of: {', '.join(OUTCOMES)}), "
    "directions (list of {party, action, deadline}; "
    f"party is one of: {', '.join(PARTIES)}; deadline is a string or null), "
    "summary (one plain-English sentence, at most 40 words). "
    "Use null when the order does not say. Do not invent dates."
)


def clip(text, n=MAX_CHARS):
    """Keep the head and the tail: the case number is at the top, the operative
    directions and the next date are usually at the bottom."""
    text = text.strip()
    if len(text) <= n:
        return text
    head = int(n * 0.45)
    tail = n - head - 20
    return text[:head].rstrip() + "\n[...]\n" + text[-tail:].lstrip()


def user_msg(order_text):
    return "Court order:\n\n" + clip(order_text)


def messages(order_text, answer_json=None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(order_text)}]
    if answer_json is not None:
        m.append({"role": "assistant", "content": answer_json})
    return m
