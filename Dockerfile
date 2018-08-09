# Grab the tensorflow image
FROM tensorflow/tensorflow:latest-py3

#navigate to the working dir
WORKDIR /app

#clone the python project
CMD ["git", "clone https://github.com/danagain/nn.git"]

CMD ["echo", "pwd"]

CMD ["cd", "/app/nn"]

CMD ["pip", "install -r requirements.txt"]

CMD ["python", "r.py"]


