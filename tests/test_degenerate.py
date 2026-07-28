import pytest
from sapphire.degenerate import is_degenerate_output


class TestDegenerateOutput:
    def test_empty_string(self):
        assert is_degenerate_output("") is True
        assert is_degenerate_output("   ") is True

    def test_single_character(self):
        assert is_degenerate_output("a") is True
        assert is_degenerate_output(" x ") is True

    def test_short_no_whitespace_no_punct(self):
        assert is_degenerate_output("abcdef") is True

    def test_short_with_whitespace_is_fine(self):
        assert is_degenerate_output("ab cd") is False

    def test_short_with_period(self):
        assert is_degenerate_output("hello!") is False
        assert is_degenerate_output("ok.") is False
        assert is_degenerate_output("yes?") is False

    def test_long_text_is_not_degenerate(self):
        assert is_degenerate_output("hello world how are you doing today?") is False

    def test_at_boundary_whitespace(self):
        assert is_degenerate_output("hello world") is False

    def test_short_with_punctuation(self):
        assert is_degenerate_output("hmm...") is False
        assert is_degenerate_output("ok!") is False

    def test_short_no_whitespace_exactly_14(self):
        assert is_degenerate_output("a" * 14) is True

    def test_stripped_whitespace_counts_as_no_whitespace(self):
        assert is_degenerate_output("a" * 14 + " ") is True
