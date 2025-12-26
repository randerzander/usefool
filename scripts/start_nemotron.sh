/home/dev/projects/llama.cpp/build/bin/llama-server -m /home/dev/projects/models/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
    -fa on \
    -c 32684 \
    -ctk q8_0 -ctv q8_0 \
    --n-gpu-layers 1000 --host 0.0.0.0 \
    --threads 24 --ubatch-size 2048 --jinja \
    -ctk q8_0 -ctv q8_0 --top-p 1 --temp 1
