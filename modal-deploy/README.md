# Serverless Deployment with Modal
This readme describes the steps for serverless model deployement with Modal platform and vLLM framework. The detailed documentation is available at [here](https://modal.com/docs/examples/vllm_inference#build-a-vllm-engine-and-serve-it).

Compares to our current deployment platform Runpod:
| Feature | **RunPod (current)** | **Modal (new)** |
| :--- | :--- | :--- |
| **Support  audio vLLM (e.g. Ultravox)** | No | **Yes** |
| **Costs (A100-80GB)** | $0.00076 / s |  **$0.00069 / s** |
| **GPU availability** | low when using network volumes |  **high**  |
| **Deployment methods** | Docker container | **single python script** |
| **serverless cold start time** | 2-3 mins | 2-3 mins |

## Deployment steps
1. register an account in https://modal.com/

2. add your HuggingFace secret in https://modal.com/secrets

3. install the Modal Python package, and create an API token.
```
pip install modal
modal setup
```

4. `vllm_inference.py` contains all the configuration for a deployment. Here are some important values that you should consider to modify:
   - `uv_pip_install`: python packges required
   - `MODEL_NAME`: model name in HuggingFace
   - `app = modal.App`:  deployed model name in Modal platform
   - `gpu=f"A100-80GB:{N_GPU}"`: the GPU type and number for deployment
   - `scaledown_window`: how long should the instance stay up with no requests?
   - `modal.Secret.from_name`: update your HuggingFace secret name if it is different.
   - `def serve()`: update the vLLM commands if necessary

5. run `modal deploy vllm_inference.py` to deploy the model to Modal platform. You can view the deployment on https://modal.com/apps

6. test the deployed model with the client script, for example:
```
python client.py \
    --app-name Sunflower32b-Ultravox  \
    --prompt "Translate to English: " \
    --audio_file "../sunflower-ultravox-vllm/audios/context_eng_1.wav"
```