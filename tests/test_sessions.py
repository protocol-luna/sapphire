import time
import pytest
from sapphire.sessions import Session, SessionStore


class TestSession:
    def test_creates_with_system_prompt(self):
        s = Session("you are a bot")
        assert len(s.messages) == 1
        assert s.messages[0] == {"role": "system", "content": "you are a bot"}

    def test_last_used_is_recent(self):
        before = time.time()
        s = Session("system")
        assert s.last_used >= before
        assert s.last_used <= time.time()


class TestSessionStore:
    @pytest.fixture
    def store(self):
        return SessionStore(
            system_prompt="you are a bot",
            ttl_seconds=3600,
            n_slots=8,
            max_history=3,
        )

    def test_get_or_create_creates_new(self, store):
        session = store.get_or_create("conv1")
        assert session.messages[0]["content"] == "you are a bot"
        assert "conv1" in store._sessions

    def test_get_or_create_reuses_existing(self, store):
        s1 = store.get_or_create("conv1")
        s2 = store.get_or_create("conv1")
        assert s1 is s2

    def test_append_user_message_creates_session(self, store):
        session = store.append_user_message("conv1", "Alice", "hello")
        assert len(session.messages) == 2
        assert session.messages[1] == {"role": "user", "content": "Alice: hello"}

    def test_append_user_message_without_username(self, store):
        session = store.append_user_message("conv1", "", "hello")
        assert session.messages[1]["content"] == "hello"

    def test_append_assistant_message(self, store):
        store.append_user_message("conv1", "Alice", "hello")
        store.append_assistant_message("conv1", "hi there")
        session = store.get_or_create("conv1")
        assert len(session.messages) == 3
        assert session.messages[2] == {"role": "assistant", "content": "hi there"}

    def test_append_assistant_to_unknown_session_does_nothing(self, store):
        store.append_assistant_message("nonexistent", "hi")
        assert "nonexistent" not in store._sessions

    def test_trim_keeps_max_exchanges(self, store):
        for i in range(8):
            store.append_user_message("conv1", "Alice", f"msg{i}")
            store.append_assistant_message("conv1", f"resp{i}")
        session = store.get_or_create("conv1")
        assert len(session.messages) <= 1 + store.max_history * 2

    def test_slot_for_is_consistent(self, store):
        assert store.slot_for("conv1") == store.slot_for("conv1")

    def test_slot_for_is_bounded(self, store):
        for sid in ["conv1", "conv2", "a" * 100, "", "abc:123"]:
            assert 0 <= store.slot_for(sid) < store.n_slots

    def test_cleanup_stale_removes_expired(self, store):
        store.get_or_create("conv1")
        store._sessions["conv1"].last_used = 0
        cleaned = store.cleanup_stale()
        assert cleaned == 1
        assert "conv1" not in store._sessions

    def test_cleanup_stale_preserves_fresh(self, store):
        store.get_or_create("conv1")
        cleaned = store.cleanup_stale()
        assert cleaned == 0
        assert "conv1" in store._sessions

    def test_reset_specific(self, store):
        store.get_or_create("conv1")
        store.get_or_create("conv2")
        store.reset("conv1")
        assert "conv1" not in store._sessions
        assert "conv2" in store._sessions

    def test_reset_all(self, store):
        store.get_or_create("conv1")
        store.get_or_create("conv2")
        store.reset()
        assert len(store._sessions) == 0

    def test_trim_within_limit_does_not_cut(self, store):
        store.max_history = 10
        for i in range(3):
            store.append_user_message("conv1", "Alice", f"msg{i}")
            store.append_assistant_message("conv1", f"resp{i}")
        session = store.get_or_create("conv1")
        assert len(session.messages) == 1 + 3 * 2

    def test_slot_distribution(self, store):
        slots = set()
        for i in range(100):
            slots.add(store.slot_for(f"conv{i}"))
        assert len(slots) >= 2
