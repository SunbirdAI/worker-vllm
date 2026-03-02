import requests
import time
import sys

# Replace with your production URL or the dev URL from `modal serve`
URL = "https://sb-modal-ws--spark-tts-salt-job-queue-fastapi-app.modal.run"

def main():
    print(f"Submitting task to {URL}...")
    try:
        resp = requests.post(
            f"{URL}/submit",
            params={
                "text": "This is a test of the asynchronous TTS system.",
                "speaker_id": 248,
                "temperature": 0.6
            }
        )
        resp.raise_for_status()
    except Exception as e:
        print("Failed to submit job:", e)
        if 'resp' in locals():
            print("Response:", resp.text)
        sys.exit(1)

    data = resp.json()
    call_id = data.get("call_id")
    if not call_id:
        print("Error: No call_id received in response:", data)
        sys.exit(1)

    print(f"✅ Job submitted successfully! Call ID: {call_id}")
    print("Polling for result...")

    # Wait for the job to complete
    max_retries = 30  # Timeout after ~150 seconds
    for i in range(max_retries):
        time.sleep(5)
        try:
            res = requests.get(f"{URL}/result/{call_id}")
            
            if res.status_code == 200:
                print("\n✅ Successfully generated audio!")
                output_file = "output.wav"
                with open(output_file, "wb") as f:
                    f.write(res.content)
                print(f"Saved to {output_file} (Size: {len(res.content)} bytes)")
                break
                
            elif res.status_code == 202:
                status = res.json().get("status", "unknown")
                print(f"[{i*5}s] Job is currently: {status}...")
                
            elif res.status_code == 404:
                print("\n❌ Job expired or not found.")
                break
                
            else:
                print(f"\n❌ Unexpected error response ({res.status_code}):", res.text)
                break
                
        except Exception as e:
            print(f"\n❌ Error connecting to server:", e)
            break
    else:
        print("\n⏳ Polling timed out. The job may still be running.")

if __name__ == "__main__":
    main()
