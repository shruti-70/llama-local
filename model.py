import requests, json
# import ollama
import gradio as gr

model = 'llama3:latest' 
context = [] 


def generate(prompt, context):
    r = requests.post('http://localhost:11434/api/generate',
                     json={
                         'model': model,
                         'prompt': prompt,
                         'context': context
                         
                     },
                     stream=False)
    r.raise_for_status()

 
    response = ""  

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        print(response_part)
        if 'error' in body:
            raise Exception(body['error'])

        response += response_part

        if body.get('done', False):
            context = body.get('context', [])
            return response, context


def chat(input, chat_history):

    chat_history = chat_history or []

    global context
    output, context = generate(input, context)

    chat_history.append((input, output))

    return chat_history, chat_history

block = gr.Blocks()


with block:

    gr.Markdown("""<h1><center> Llama 3 </center></h1>
    """)

    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Type here")

    state = gr.State()
   

    submit = gr.Button("SEND")

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])


block.launch(debug=True)