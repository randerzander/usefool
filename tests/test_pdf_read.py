from tools.read_url import read_url

# Test with a known PDF URL
test_pdf_url = "https://arxiv.org/pdf/2408.09869v4"
print(f"Testing PDF read from: {test_pdf_url}")
content = read_url(test_pdf_url)

if content and not content.startswith("Error"):
    print(f"Success! Content length: {len(content)}")
    print("First 500 characters:")
    print(content[:500])
else:
    print(f"Failed: {content}")
