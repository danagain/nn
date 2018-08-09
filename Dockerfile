FROM python:3

WORKDIR /var/app

COPY . .

RUN pip install -r requirements.txt

CMD ["python","-u","r.py"]
