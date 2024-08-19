import sys
import io

def execute_code(code):
    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(code, {})
    except Exception as e:
        print(f"Error executing code: {e}")
    finally:
        sys.stdout = old_stdout

    output = new_stdout.getvalue()
    return output
