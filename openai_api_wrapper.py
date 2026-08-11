from __future__ import annotations

try:
    import openai
except ImportError:  # optional dependency: pip install openai
    openai = None

def correct_code_with_llm(code, error_message):
    prompt = f"""
    The following code has an error:

    ```python
    {code}
    ```

    Error Message:
    {error_message}

    Please provide a corrected version of the code.
    """

    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=200,
        temperature=0
    )

    corrected_code = response.choices[0].text.strip()
    return corrected_code
