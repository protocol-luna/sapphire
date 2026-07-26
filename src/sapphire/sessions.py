import time


class Session:
    __slots__ = ("messages", "last_used")

    def __init__(self, system_prompt: str):
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.last_used: float = time.time()


class SessionStore:
    def __init__(self, system_prompt: str, ttl_seconds: float, n_slots: int, max_history: int):
        self.system_prompt = system_prompt
        self.ttl_seconds = ttl_seconds
        self.n_slots = n_slots
        self.max_history = max_history
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(self.system_prompt)
            self._sessions[session_id] = session
        session.last_used = time.time()
        return session

    def append_user_message(self, session_id: str, username: str, text: str) -> Session:
        session = self.get_or_create(session_id)
        content = f"{username}: {text}" if username else text
        session.messages.append({"role": "user", "content": content})
        return session

    def append_assistant_message(self, session_id: str, text: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.messages.append({"role": "assistant", "content": text})
        self._trim(session)

    def _trim(self, session: Session) -> None:
        exchanges = session.messages[1:]
        if len(exchanges) > self.max_history * 2:
            session.messages = [session.messages[0], *exchanges[-self.max_history * 2:]]

    def slot_for(self, session_id: str) -> int:
        h = 0
        for ch in session_id:
            h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
        return abs(h) % self.n_slots

    def cleanup_stale(self) -> int:
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self.ttl_seconds
        ]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    def reset(self, session_id: str | None = None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)
        else:
            self._sessions.clear()
