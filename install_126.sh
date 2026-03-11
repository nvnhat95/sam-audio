pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip uninstall -y xformers
pip install --no-deps xformers==0.0.33.post2
pip uninstall -y torchcodec
pip install torchcodec==0.8.1
pip install uvicorn fastapi python-multipart

## might need
apt-get update
apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev