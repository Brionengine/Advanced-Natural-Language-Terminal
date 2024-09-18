from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_code', methods=['POST'])
def generate_code():
    data = request.get_json()
    user_input = data['message']
    
    # Call the LLM as before
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=f"Convert the following description into Python code:\n\n{user_input}",
        max_tokens=150
    )
    
    code = response.choices[0].text.strip()
    return jsonify({'code': code})

if __name__ == '__main__':
    app.run(debug=True)
