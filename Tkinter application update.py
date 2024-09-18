import tkinter as tk
from tkinter import scrolledtext
import openai

class BrionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brion Assistant")
        self.root.geometry("600x600")
        
        self.conversation = []
        
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
        
        brion = Brion()
        response = brion.generate_code(user_input)
        
        self.output_box.insert(tk.END, f"Assistant:\n{response}\n\n", 'code')
        self.output_box.tag_config('code', font=('Courier', 10), foreground='blue')
        
        # Optionally, execute the code
        # self.execute_generated_code(response)
        
class Brion:
    def __init__(self):
        # Initialize conversation history
        self.conversation = []
        
    def generate_code(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})
        
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.conversation]
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        
        assistant_message = response.choices[0].message["content"]
        self.conversation.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

if __name__ == "__main__":
    root = tk.Tk()
    app = BrionApp(root)
    root.mainloop()
