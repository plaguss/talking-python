# app/Dockerfile
# Use for starters: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
FROM python:3.10-slim

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
