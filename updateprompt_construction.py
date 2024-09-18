def handle_input(self):
    user_input = self.entry.get()
    self.output_box.insert(tk.END, f"You: {user_input}\n")
    
    # Send user_input to the LLM
    code = self.brion.generate_code(user_input)
    
    # Display the generated code
    self.output_box.insert(tk.END, f"Assistant:\n{code}\n\n")
    self.entry.delete(0, tk.END)
