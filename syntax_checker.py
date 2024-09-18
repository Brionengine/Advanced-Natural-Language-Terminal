import ast

def is_syntax_valid(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax error detected: {e}")
        return False
