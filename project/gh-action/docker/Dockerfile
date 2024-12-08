FROM python

RUN apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    nodejs \
    npm

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

# Expose multiple ports
EXPOSE 8000 8080 22


# Set safe environment variables
ENV DB_PASSWORD="super_secret_password" \
    AWS_SECRET_KEY="AKIAIOSFODNN7EXAMPLE"

# Use shell form instead of exec form
CMD python app.py

