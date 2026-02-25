@echo off
REM =============================================================================
REM prepare.bat — AI Asset Generation Pipeline (s01e01)
REM RTX 4060 8 GB VRAM / CUDA 12.1 / Python 3.10
REM
REM Run this from the repo root with the venv activated:
REM   cd C:\Users\tnnd\data\svd
REM   venv\Scripts\activate
REM   code\prepare.bat
REM =============================================================================

echo.
echo [1/5] Installing PyTorch 2.4.1 with CUDA 12.1...
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/5] Installing xformers (must use PyTorch index to match CUDA version)...
pip install xformers --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/5] Installing AudioCraft without its hard-pinned av==11.0.0 dependency...
echo       (av==11.0.0 has no Windows binary wheel; we use 12.0.0 instead)
pip install audiocraft --no-deps
pip install "av==12.0.0" --only-binary=av
pip install encodec flashy num2words spacy tqdm librosa

echo.
echo [3b/5] Downgrading NumPy to 1.x (torch 2.4.x C extensions are not compatible with NumPy 2.x)...
pip install "numpy<2"

echo [4/5] Installing remaining requirements...
pip install -r code\requirements.txt

echo.
echo [4b/5] Installing Real-ESRGAN and basicsr (upscaling — gen_upscale.py)...
echo        If this fails, retry with: pip install realesrgan basicsr --no-build-isolation
pip install realesrgan basicsr

echo.
echo [5/5] Done. Quick sanity check:
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import diffusers; print('diffusers:', diffusers.__version__)"
python -c "import kokoro; print('kokoro: OK')"
python -c "from audiocraft.models import MusicGen; print('audiocraft: OK')"

echo.
echo =============================================================================
echo All packages installed. Recommended test run order:
echo    1. python code\gen_character_images.py
echo    2. python code\gen_background_images.py
echo    3. python code\gen_background_video.py
echo    4. python code\gen_character_mattes.py
echo    5. python code\gen_upscale.py
echo    6. python code\gen_character_animation.py
echo    7. python code\gen_lipsync.py
echo    8. python code\gen_tts.py
echo    9. python code\gen_sfx.py
echo   10. python code\gen_music.py
echo =============================================================================
