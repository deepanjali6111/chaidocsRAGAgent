FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "10000", "--server.address", "0.0.0.0"]
