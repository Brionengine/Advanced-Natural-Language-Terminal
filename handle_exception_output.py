def execute_generated_code(self, code):
    try:
        exec_locals = {}
        exec(code, {}, exec_locals)
        output = exec_locals.get('output', '')
        self.output_box.insert(tk.END, f"Execution Output:\n{output}\n\n")
    except Exception as e:
        self.output_box.insert(tk.END, f"Error during execution: {e}\n\n")
