# Use for starters: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
FROM python:3.10-slim

# To install hnswlib
RUN apt-get update && apt-get install build-essential -y

RUN python -m pip install --upgrade pip

# Copy and install dependencies to work with chroma and related content
COPY /src /tmp/src
RUN python -m pip install /tmp/src

# Copy the streamlit app content and install it.
COPY /app /app/
WORKDIR /app/
RUN python -m pip install .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
