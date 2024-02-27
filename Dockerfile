# Specify the base image
FROM python:3.8.9

# Set the working directory
WORKDIR /app

# Copy the project files to the working directory
COPY tmp /app/tmp
COPY app.py /app/app.py
COPY functions.py /app/functions.py
COPY requirements.txt /app/requirements.txt

# Install the required packages
# RUN pip3 install --upgrade pip

RUN pip3 install --timeout 10000 -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
