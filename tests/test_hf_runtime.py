import unittest

from kl_anchors.hf_runtime import render_chat_prompt


class PlainTokenizer:
    pass


class ChatTokenizer:
    chat_template = "configured"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "rendered"


class RenderChatPromptTests(unittest.TestCase):
    def test_missing_chat_template_attribute_uses_plain_text(self):
        tokenizer = PlainTokenizer()
        self.assertEqual(render_chat_prompt(tokenizer, "question"), "question")
        self.assertEqual(
            render_chat_prompt(tokenizer, "question", " system "),
            "system\n\nquestion",
        )

    def test_configured_chat_template_receives_system_and_user_messages(self):
        tokenizer = ChatTokenizer()
        self.assertEqual(render_chat_prompt(tokenizer, "question", "system"), "rendered")
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
