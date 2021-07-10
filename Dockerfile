FROM python:3.8.11-alpine3.14

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["cd", "app/"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]