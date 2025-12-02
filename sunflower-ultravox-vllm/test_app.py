import requests

# url = "http://localhost:8001/transcribe"
# url = "http://69.143.221.78:23779/transcribe"
url ="http://4.151.151.100:8001/transcribe"

with open("audios/context_eng_7.wav", "rb") as f:
    files = {"audio_file": f}
    data = {
        "task": "Translate to English: ",
        "temperature": 0.1
    }
    
    response = requests.post(url, files=files, data=data, stream=True)
    
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            print(chunk, end='', flush=True)