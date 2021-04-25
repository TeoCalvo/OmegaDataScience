FROM python:3

COPY . .

RUN pip install -r requirements.txt

CMD tail -f /dev/null