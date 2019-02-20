FROM tensorflow/tensorflow:1.12.0-py3

MAINTAINER Sebastien Soudan "sebastien.soudan@gmail.com"

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --no-cache -r requirements.txt

COPY ./src/main/python/ /app/
COPY ./src/main/resources/ /app/
COPY check.sh /app/
COPY test.sh /app/
COPY pylama.ini /app/

RUN ./check.sh
RUN ./test.sh

VOLUME /app/output
VOLUME /app/data

ENTRYPOINT [ "python3" ]

CMD [ "run_tfp.py" ]
