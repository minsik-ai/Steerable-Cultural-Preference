"""Real tiny Llama/PEFT integration for the chat profile."""
from tests.test_original_tulu_smoke import OriginalTuluSmokeTests


class ChatTuluSmokeTests(OriginalTuluSmokeTests):
    chat_profile = True
