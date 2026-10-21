from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext

from brion import Brion

class BrionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brion Assistant")
        self.root.geometry("600x600")
        
        # One session for the window's lifetime, so follow-ups keep context.
        self.brion = Brion()
        
        self.output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=30)
        self.output_box.pack(pady=10)
        
        self.entry = tk.Entry(root, width=70)
        self.entry.pack(pady=5)
        
        self.send_button = tk.Button(root, text="Send", command=self.handle_input)
        self.send_button.pack(pady=5)
        
    def handle_input(self):
        user_input = self.entry.get()
        self.output_box.insert(tk.END, f"You: {user_input}\n")
        self.entry.delete(0, tk.END)
        
        response = self.brion.generate_code(user_input)
        
        self.output_box.insert(tk.END, f"Assistant:\n{response}\n\n", 'code')
        self.output_box.tag_config('code', font=('Courier', 10), foreground='blue')
        
        # Optionally, execute the code
        # self.execute_generated_code(response)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = BrionApp(root)
    root.mainloop()
