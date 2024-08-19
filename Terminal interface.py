def start_terminal():
    print("Welcome to your advanced terminal. Type your instructions in plain English.")
    while True:
        user_input = input(">> ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        code = parse_natural_language(user_input)
        if code:
            print(f"Generated Code:\n{code}")
            save_code(user_input, code)
            output = execute_code(code)
            print(f"Execution Output:\n{output}")
        else:
            print("Sorry, I couldn't generate code for that input.")
