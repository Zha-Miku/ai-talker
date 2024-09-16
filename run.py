import gradio as gr
import ollama

from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os
import requests

import pygame


# os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

# link to ollama by langchain
llm = ChatOllama(base_url="http://localhost:11434",
                 model="qwen:0.5b",
                 #  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                 )

# from audio get text by fast-whisper


def get_text(file):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    from faster_whisper import WhisperModel

    model_size = "small"
    # path = r"D:\Project\Python_Project\FasterWhisper\large-v3"
    path = './apply-ollama/whisp-model'

    # Run on GPU with FP16
    model = WhisperModel(model_size_or_path=path,
                         device="cuda", local_files_only=True)

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # ???????????????????????????????????????????????????????????????????????????????????????????????
    segments, info = model.transcribe(file, beam_size=5, language="en",
                                      vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))

    # print("Detected language '%s' with probability %f" %
    #       (info.language, info.language_probability))

    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" %
    #           (segment.start, segment.end, segment.text))
    t = ''
    for s in segments:
        t += s.text
    return t  # pointing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# link ollama api by ollama lib


def ollama_resp():
    import ollama

    host = "127.0.0.1"
    port = "11434"
    client = ollama.Client(host=f"http://{host}:{port}")
    res = client.chat(model="qwen:0.5b", messages=[
                      {"role": "user", "content": "介绍大语言模型"}], options={"temperature": 0})
    return res

# add history by langchain and format use in chatInterface


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    yield gpt_response.content


# 初始化Ollama客户端
host = "127.0.0.1"
port = "11434"
client = ollama.Client(host=f"http://{host}:{port}")

# Use ollama api


def get_response(user_input):
    # 使用Ollama API发送请求
    res = client.chat(
        model="qwen:0.5b",
        messages=[{"role": "user", "content": user_input}],
        options={"temperature": 0}
    )
    return res['message']['content']
    # return res


# create audio by gsv


def resp_audio(prompt):
    url = "http://127.0.0.1:9880"
    json = {
        "refer_wav_path": "cyse.wav",
        "prompt_text": "Can you speak english?",
        "prompt_language": "en",
        "text": prompt,
        "text_language": "en"
    }
    resp = requests.post(url=url, json=json)
    return resp.content

# play audio


def resp_and_play_audio(prompt):  # gen by glm
    import pygame
    from pydub import AudioSegment
    import io

    # 假设 audio_bytes 是音频文件的字节流
    audio_bytes = resp_audio(prompt=prompt)

    # 使用 pydub 读取音频数据
    audio_segment = AudioSegment.from_file(
        io.BytesIO(audio_bytes), format="wav")

    # 检查音频参数
    print(audio_segment.frame_rate, audio_segment.channels)

    # 转换为 pygame 可以播放的格式
    pygame.mixer.init(frequency=audio_segment.frame_rate,
                      channels=audio_segment.channels)
    audio_segment = audio_segment.set_channels(2)  # 确保是双声道
    audio_segment = audio_segment.set_frame_rate(audio_segment.frame_rate)

    # 保存到临时文件或字节流
    temp_file = io.BytesIO()
    audio_segment.export(temp_file, format="wav")
    temp_file.seek(0)

    # 加载音频数据到 pygame
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()

    # 等待音乐播放完毕
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# UI set
with gr.Blocks(
    title="Novel WebUI",
    theme='soft'

) as iface:
    gr.Markdown(
        value=(
            'AI Talker\n' +
            'abc.'
        )
    )
    with gr.TabItem('0-one-talk'):
        gr.Markdown(
            value=(
                'Once talk and no memory'
            )
        )
        inp = gr.Textbox(
            # label='inpp'
        )
        outp = gr.Textbox(
            label='Competiton'
        )
        run = gr.Button(
            value='Run',
            scale=1,
            visible=True
        )
        run.click(get_response, [inp], [outp])
    with gr.TabItem('1-many-talk'):
        gr.Markdown(
            value=(
                'talk and some memory'
            )
        )
    with gr.TabItem('2-chatbox'):
        chatbot = gr.Chatbot(
            height=300, placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything"),
        textbox = gr.Textbox(
            placeholder="Ask me a yes or no question", container=False, scale=7),
        title = "Yes Man",
        description = "Ask Yes Man any question",
        theme = "soft",
        examples = ["Hello", "Am I cool?", "Are tomatoes vegetables?"],
        cache_examples = True,
        retry_btn = None,
        undo_btn = "Delete Previous",
        clear_btn = "Clear",
    with gr.TabItem('3-gradio:chatinterface-longchain'):
        a = []
        gr.ChatInterface(
            predict,
            examples=['Hi', 'Hello'],
            show_progress=['full'],
        )
        g_btn = gr.Button('Get his')

    with gr.TabItem('4-gsv'):
        gr.Markdown(
            value='this is demo'
        )
        audio = gr.Audio(
            value='./refer.wav'
        )
        prompt = gr.Textbox(
            value='',
            label='Inp Text'
        )
        out_audio = gr.Audio()
        get = gr.Button(
            value='get it!'
        )
        get.click(resp_audio, [prompt], [out_audio])
    with gr.TabItem('5-whisp'):
        gr.Markdown(
            value='get text from audio'
        )
        gr.Chatbot(

        )
    with gr.TabItem('6-chatbox-textbox-langchain'):
        gr.Markdown(
            value='more cunstom'
        )
        cb = gr.Chatbot(
            type='messages'
        )
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}]

        def resp_qwen(messg, audio):
            from dashscope import Generation
            import dashscope
            import os
            text = ''
            if audio:
                text = get_text(audio)
                messg = text

            dashscope.api_key = 'sk-b44144ea6bd64afcb88e2e2dd374df09'

            def get_response(messages):
                response = Generation.call(model="qwen-turbo",
                                           messages=messages,
                                           #    api_key=os.getenv("sk-b44144ea6bd64afcb88e2e2dd374df09"),
                                           # 将输出设置为"message"格式
                                           result_format='message')
                return response

            # 您可以自定义设置对话轮数，当前为3

            # user_input = input("请输入：")
            user_input = messg
            # 将用户问题信息添加到messages列表中
            messages.append({'role': 'user', 'content': user_input})
            assistant_output = get_response(
                messages).output.choices[0]['message']['content']
            # 将大模型的回复信息添加到messages列表中
            messages.append({'role': 'assistant', 'content': assistant_output})
            resp_and_play_audio(assistant_output)
            yield '', messages

        def predict1(message, history):  # use langchain
            history_langchain_format = []
            for human, ai in history:
                history_langchain_format.append(HumanMessage(content=human))
                history_langchain_format.append(AIMessage(content=ai))
            history_langchain_format.append(HumanMessage(content=message))
            gpt_response = llm(history_langchain_format)
            return gpt_response.content
        hist = []
        # importent! format msg

        def cbot(msgs):
            resp = predict1(msgs, hist)
            hist.append({'role': 'user', 'content': msgs})
            hist.append({'role': 'assistant', 'content': resp})
            # audio = resp_audio(resp)
            # play_sound()
            resp_and_play_audio(resp)
            return '', hist

        def send(audio):

            text = get_text(audio)
            msgs = resp_qwen(text)
            return msgs[0], msgs[1]

        with gr.Row():
            msg = gr.Textbox(type='text')
            inp_au = gr.Audio(
                type='filepath'
            )
            up_btn = gr.Button(
                value='send audio'
            )
        cl = gr.Button('Clear')
        # au = gr.Audio()
        # fn= resp_qwen(qwen-turbo) or cbot(ollama)
        msg.submit(resp_qwen, [msg, inp_au], [msg, cb])
        up_btn.click(resp_qwen, [msg, inp_au], [msg, cb])
    with gr.TabItem(label='7-get-audio-by-whisper'):
        with gr.Row():
            inp_ref = gr.Audio(
                label=("请上传3~10秒内参考音频，超过会报错！"),
                type="filepath"
            )
            with gr.Column():
                ref_text_free = gr.Checkbox(label=(
                    "开启无参考文本模式。不填参考文本亦相当于开启。"), value=False, interactive=True, show_label=True)
                gr.Markdown(
                    ("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。"))
                prompt_text = gr.Textbox(label=("参考音频的文本"), value="")
            prompt_language = gr.Dropdown(
                label=("参考音频的语种"), choices=[("中文"),  ("英文"),  ("日文"),  ("中英混合"),  ("日英混合"),  ("多语种混合")], value=("中文")
            )
        gr.Markdown(value=("*请填写需要合成的目标文本和语种模式"))


# 启动Gradio应用
iface.launch()
