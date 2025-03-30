# Use a multi-stage build to reduce final image size
FROM python:3.10-slim AS builder

# Set environment variables to reduce Python overhead
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only requirements file first
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Use tensorflow-cpu to save space
RUN pip install -r requirements.txt

# Second stage - minimal runtime image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only download the specific NLTK data we need
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

# Copy only the required files for deployment
COPY deployment/ /app/deployment/
COPY models/ /app/models/
COPY run.py /app/

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Command to run the services
CMD ["python", "run.py"]