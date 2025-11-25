import requests

url = "http://localhost:8001/transcribe"

with open("audios/context_eng_1.wav", "rb") as f:
    files = {"audio_file": f}
    data = {
        "task": "Translate to English: ",
        "temperature": 0.1
    }
    
    response = requests.post(url, files=files, data=data, stream=True)
    
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            print(chunk, end='', flush=True)