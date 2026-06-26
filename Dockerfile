FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

ENV PYTHONPATH=/app
ENV TORCHDYNAMO_DISABLE=1
ENV TORCH_COMPILE_DISABLE=1

RUN grep -v "pywin32\|scikit-sparse" requirements.txt > requirements_linux.txt && \
    pip install --no-cache-dir -r requirements_linux.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir diffusers==0.38.0

RUN pip install --no-cache-dir einops==0.8.2

RUN pip install --no-cache-dir matplotlib==3.10.6

RUN python -c "from OAT.Models import NFAE, CTOPUNet; NFAE.from_pretrained('OpenTO/NFAE'); CTOPUNet.from_pretrained('OpenTO/LDM')"



EXPOSE 8000

CMD ["python", "maker.py"]