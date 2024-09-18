import tkinter as tk
from tkinter import scrolledtext
from brion import Brion

class BrionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brion Assistant")
        self.root.geometry("600x600")

        self.output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=30)
        self.output_box.pack(pady=10)

        self.entry = tk.Entry(root, width=70)
        self.entry.pack(pady=5)
        self.entry.bind('<Return>', lambda event: self.handle_input())

        self.send_button = tk.Button(root, text="Send", command=self.handle_input)
        self.send_button.pack(pady=5)

        self.brion = Brion()

    def handle_input(self):
        user_input = self.entry.get()
        if not user_input.strip():
            return
        self.output_box.insert(tk.END, f"You: {user_input}\n")
        self.entry.delete(0, tk.END)

        code = self.brion.generate_code(user_input)
        self.output_box.insert(tk.END, f"Assistant:\n{code}\n\n", 'code')
        self.output_box.tag_config('code', font=('Courier', 10), foreground='blue')

        # Attempt to execute the code
        self.execute_generated_code(code)

    def execute_generated_code(self, code):
        self.output_box.insert(tk.END, "Executing generated code...\n")
        try:
            exec_locals = {}
            exec(code, {}, exec_locals)
            output = exec_locals.get('output', '')
            if output:
                self.output_box.insert(tk.END, f"Execution Output:\n{output}\n\n")
            else:
                self.output_box.insert(tk.END, "Code executed successfully.\n\n")
        except Exception as e:
            self.output_box.insert(tk.END, f"Error during execution: {e}\n\n", 'error')
            self.output_box.tag_config('error', foreground='red')

            # Attempt to correct the code
            self.output_box.insert(tk.END, "Attempting to correct the code...\n")
            corrected_code = self.brion.correct_code(code, str(e))
            self.output_box.insert(tk.END, f"Assistant Corrected Code:\n{corrected_code}\n\n", 'code')
            self.output_box.tag_config('code', font=('Courier', 10), foreground='green')

            # Re-attempt execution with corrected code
            self.output_box.insert(tk.END, "Re-executing corrected code...\n")
            try:
                exec_locals = {}
                exec(corrected_code, {}, exec_locals)
                output = exec_locals.get('output', '')
                if output:
                    self.output_box.insert(tk.END, f"Execution Output:\n{output}\n\n")
                else:
                    self.output_box.insert(tk.END, "Corrected code executed successfully.\n\n")
            except Exception as e:
                self.output_box.insert(tk.END, f"Corrected code failed: {e}\n\n", 'error')

if __name__ == "__main__":
    root = tk.Tk()
    app = BrionApp(root)
    root.mainloop()
