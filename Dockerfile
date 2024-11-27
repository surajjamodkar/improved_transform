FROM python:3.9

WORKDIR /app

COPY ./src .

RUN pip install -r requirements.txt

CMD ["python", "run_gmm.py"]
