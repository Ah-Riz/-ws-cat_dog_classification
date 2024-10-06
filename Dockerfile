FROM python:3.9.12

WORKDIR /app
    
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install virtualenv
CMD ["virtualenv", "env"]
CMD ["source", "env/bin/activate"]
RUN pip install grpcio
RUN pip install --upgrade google-cloud-firestore
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]