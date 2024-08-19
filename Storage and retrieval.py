import json

code_database = {}

def save_code(command, code):
    code_database[command] = code
    with open('code_database.json', 'w') as f:
        json.dump(code_database, f)

def retrieve_code(command):
    if not code_database:
        try:
            with open('code_database.json', 'r') as f:
                code_database.update(json.load(f))
        except FileNotFoundError:
            return "No such command found."
    return code_database.get(command, "No such command found.")
