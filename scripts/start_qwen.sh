/home/dev/projects/llama.cpp/build/bin/llama-server -m /home/dev/projects/models/Qwen3-VL-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
    -fa on -c 128000 -ctk q8_0 -ctv q8_0 \
    --n-gpu-layers 1000 --host 0.0.0.0 --threads 24 --ubatch-size 2048 \
    --jinja -ctk q8_0 -ctv q8_0 --top-p 0.8 --top-k 20 --temp 0.7 --min-p 0.0 \
    --presence-penalty 1.5 --mmproj /home/dev/projects/models/mmproj-F32.gguf
