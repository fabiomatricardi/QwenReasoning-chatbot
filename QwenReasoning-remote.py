import gradio as gr
from openai import OpenAI
import random
import string
import tiktoken
import datetime
# Multiple choice of Google Models from OpenTouter AI
# One API key to rule them all
# a complex dictionary with icons, model name for the dorpdown, NCTX modalities
# TO be DOne:
# - create a logfile for the chat
# - LOG the changes in Model and truncation of the Chat messages in case of MAX LENGHT reached


def countHistoryTokens(chatmessages):
    """
    Use tiktoken to count the number of tokens
    chatmessages -> list of dict in ChatML format
    Return -> int number of tokens counted
    """
    if (chatmessages is None) or len(chatmessages)==0:
       numoftokens = 0
    else: 
        text = ''
        for items in chatmessages:
            text += items['content'] + "\n"
        encoding = tiktoken.get_encoding("cl100k_base") 
        numoftokens = len(encoding.encode(text))
    return numoftokens

def countTokens(text):
    """
    Use tiktoken to count the number of tokens
    text -> str input
    Return -> int number of tokens counted
    """
    encoding = tiktoken.get_encoding("cl100k_base") 
    numoftokens = len(encoding.encode(text))
    return numoftokens

def writehistory(filename,text):
    """
    save a string into a logfile with python file operations
    filename -> str pathfile/filename
    text -> str, the text to be written in the file
    """
    with open(f'{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    Return -> str, the filename with n random alphanumeric charachters
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    print(f'Logfile_{res}.md  CREATED')
    return f'Logfile_{res}.md'

extrainfo = """### Model list from [openrouter.ai](https://openrouter.ai)

- google/gemma-3-4b-it:free
- google/gemma-3-27b-it:free
- google/gemma-3-12b-it:free
- google/gemma-3-1b-it:free

- google/gemini-2.0-flash-lite-preview-02-05:free
- google/gemini-2.0-flash-exp:free
- google/gemini-2.0-pro-exp-02-05:free
- google/gemini-2.0-flash-thinking-exp-1219:free
- google/learnlm-1.5-pro-experimental:free
- google/gemini-flash-1.5-8b-exp
- google/gemini-2.0-flash-thinking-exp:free


BONUS
- rekaai/reka-flash-3:free
- moonshotai/moonlight-16b-a3b-instruct:free
- mistralai/mistral-small-3.1-24b-instruct:free   128k in/out
- deepseek/deepseek-r1-distill-qwen-32b:free  16k in/out
- deepseek/deepseek-r1-distill-qwen-14b:free  64k in/out
- deepseek/deepseek-r1-distill-llama-70b:free  128k in/out

"""
note = """#### ⚠️ Remember to put your API key for Open Router
> you can find the field in the Additional Inputs<br>
> you can get an API key for free from [openrouter.ai](https://openrouter.ai/settings/keys)
<br>Starting settings: `Temperature=0.45` `Max_Length=2048`
"""

open_mod_lst = [('reka-flash-3','rekaai/reka-flash-3:free'),
                ('gemini-2.0-flash-thinking','google/gemini-2.0-flash-thinking-exp:free'),
                ('deepseek-r1-distill-qwen-14b','deepseek/deepseek-r1-distill-qwen-14b:free'),
                ('deepseek-r1-distill-llama-70b','deepseek/deepseek-r1-distill-llama-70b:free')]

with gr.Blocks(theme=gr.themes.Citrus(),fill_width=True) as demo: #gr.themes.Ocean() #https://www.gradio.app/guides/theming-guide
    gr.Markdown("# Chat with the best Google Models for free - works also in China")
    mychoice = gr.Markdown(f'#### Using model: *google/gemma-3-4b-it:free*')
    LOGfileNAME = genRANstring(5)
    with gr.Row():
        with gr.Column(scale=1):
            maxlen = gr.Slider(minimum=250, maximum=8192, value=2048, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.45, step=0.1, label="Temperature")          
            APIKey = gr.Textbox(value="", 
                        label="⚠️ Remember to put your Open Router API key",
                        type='password',placeholder='Paste your API key',)
            slectedModel = gr.Dropdown(open_mod_lst)
            def modelChange(a,b):
                """
                a is the dropdown list
                b is the chatbot_history list
                """
                open_mod_lst = {'rekaai/reka-flash-3:free':'32k',
                                'google/gemini-2.0-flash-thinking-exp:free':'1M',
                                'deepseek/deepseek-r1-distill-qwen-14b:free':'64k',
                                'deepseek/deepseek-r1-distill-llama-70b:free':'128k'}
                NCTX = open_mod_lst[a]
                usedContext = countHistoryTokens(b)
                return f'#### Using model: *{a}*',f'Context: {NCTX}<br>Output: 8k<br>Used Context: {usedContext} tok'
            
            kpi = gr.Markdown(label='KPI',value='Context: 128k<br>Output: 8k<br>Used Context: 0 tok')
            logfile = gr.Textbox(label='Log File',value=LOGfileNAME,interactive=False)
            with gr.Accordion(open=False,label='Additional info'):
                gr.Markdown(note)
                gr.Markdown(extrainfo)            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages",show_copy_button = True,
                    avatar_images=['https://i.ibb.co/PvqbDphL/user.png',
                                   'https://i.ibb.co/4ZRfPzCb/deepseek.png'],
                    height=480, layout='panel')
            msg = gr.Textbox(lines=3)
            clear = gr.ClearButton([msg, chatbot])
            slectedModel.change(modelChange,inputs=[slectedModel,chatbot],outputs=[mychoice,kpi])


    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]    

    def respond(model,chat_history, api,t,m,logfile):
        client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api,
                )
        # SUGGESTED BY cLAUDE
        # https://claude.ai/share/62edfc2f-737a-405c-ba62-1de9e383dcd6
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://thepoorgpuguy.substack.com/", 
                "X-Title": "Fabio Matricardi is The Poor GPU Guy", 
            },
            extra_body={},
            model=model,        
            messages=chat_history,
            max_tokens=m,
            stream=True,
            temperature=t)

        reasoning_content = ""
        final_content = ""

        chat_history.append({"role": "assistant", "content": ""})

        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # Use try/except to gracefully handle attribute errors
            try:
                if delta.reasoning is not None:
                    reasoning_content += delta.reasoning
                    chat_history[-1]['content'] = f"**Thinking...🤔**\n {reasoning_content}\n\n**Response**\n {final_content}"
            except AttributeError:
                # If reasoning attribute isn't available, try content
                pass
                
            try:
                if delta.content is not None:
                    final_content += delta.content
                    chat_history[-1]['content'] = f"**Thinking...🤔**\n {reasoning_content}\n\n**Response**\n {final_content}"
            except AttributeError:
                # If content attribute isn't available, continue
                pass

            yield chat_history          
        writehistory(logfile,f'USER: {chat_history[-2]["content"]}\nASSISTANT: {chat_history[-1]["content"]}\n\n')

    def KPIChange(a,b):
        """
        a is the dropdown list
        b is the chatbot_history list
        """
        open_mod_lst = {'rekaai/reka-flash-3:free':'32k',
                        'google/gemini-2.0-flash-thinking-exp:free':'1M',
                        'deepseek/deepseek-r1-distill-qwen-14b:free':'64k',
                        'deepseek/deepseek-r1-distill-llama-70b:free':'128k'}
        NCTX = open_mod_lst[a]
        usedContext = countHistoryTokens(b)
        return f'Context: {NCTX}<br>Output: 8k<br>Used Context: {usedContext} tok'


    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
                respond, [slectedModel,chatbot,APIKey,temperature,maxlen,logfile], [chatbot]).then(
                KPIChange, [slectedModel,chatbot], [kpi])

demo.launch(inbrowser=True)