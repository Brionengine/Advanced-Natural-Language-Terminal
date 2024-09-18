import openai

class Brion:
    def __init__(self):
        self.conversation = []

    def generate_code(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})

        messages = self.conversation.copy()

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )

        assistant_message = response.choices[0].message["content"]
        self.conversation.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def correct_code(self, code, error_message):
        prompt = f"""
        The following Python code contains an error:

        ```python
        {code}
        ```

        The error message is:
        {error_message}

        Please provide a corrected version of the code.

        Corrected Code:
        """

        response = openai.Completion.create(
            engine="gpt-4o",
            prompt=prompt,
            max_tokens=300,
            temperature=0
        )

        corrected_code = response.choices[0].text.strip()
        return corrected_code
