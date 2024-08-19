def parse_natural_language(input_text):
    try:
        response = openai.Completion.create(
            engine="gpt-4o",  # Using GPT-4o as the engine
            prompt=f"Translate this English text to Python code: {input_text}",
            max_tokens=150,
            temperature=0.5,
            top_p=1.0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error during NLP parsing: {e}")
        return None
