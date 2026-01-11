#!/usr/bin/env python3
"""
Test to verify the exact message format being sent to llama.cpp for vision.
"""

import os
import json
import requests
from utils import download_image, image_to_base64

# Test image URL
test_image = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/300px-Cat03.jpg'

print("Downloading and encoding image...")
img_path = download_image(test_image)
b64_img = image_to_base64(img_path)
print(f"Base64 length: {len(b64_img)}")

# Build the message in OpenAI vision format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            }
        ]
    }
]

# Try calling the API directly
api_key = os.getenv('OPENROUTER_API_KEY', 'test')
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "Qwen3-VL-30B-A3B-Instruct-UD-Q6_K_XL",
    "messages": messages,
    "max_tokens": 500
}

print("\n" + "="*80)
print("Message structure being sent:")
print("="*80)
# Print without the huge base64 data
msg_copy = json.loads(json.dumps(messages))
if msg_copy[0]["content"][1]["image_url"]["url"].startswith("data:"):
    msg_copy[0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,<{len(b64_img)} chars>"
print(json.dumps(msg_copy, indent=2))

print("\n" + "="*80)
print("Sending request to llama.cpp...")
print("="*80)

try:
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("\n" + "="*80)
        print("RESPONSE:")
        print("="*80)
        print(content)
        print("="*80)
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up
    os.remove(img_path)
