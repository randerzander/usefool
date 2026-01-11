
import logging
import sys
from tools.read_url import read_url

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url = "https://arxiv.org/pdf/2408.09869v4"
content = read_url(url)

print(f"Content length: {len(content)}")
print("First 500 characters of content:")
print(content[:500])
