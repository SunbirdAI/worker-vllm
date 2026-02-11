import argparse
import requests
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Whisper Client")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--url", type=str, required=True, help="URL of the Modal endpoint")
    parser.add_argument("--language", type=str, default=None, help="Optional language code for transcription")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}")
        sys.exit(1)

    with open(args.audio, "rb") as f:
        audio_data = f.read()

    print(f"Sending {len(audio_data)} bytes of audio data to {args.url}...")
    
    # Send audio data as raw request body
    params = {}
    if args.language:
        params["language"] = args.language

    response = requests.post(
        args.url,
        data=audio_data,
        headers={"Content-Type": "application/octet-stream"},
        params=params,
    )

    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
