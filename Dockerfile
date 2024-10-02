FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN apt-get update && apt-get install -y python3.10-venv cargo
RUN pip install poetry==1.8.3
ENV \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
WORKDIR /mist

# Configure SSH
RUN mkdir -p -m 06000 ~/.ssh
RUN apt-get install -y openssh-client
RUN mkdir -p -m 0600 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Install env
COPY pyproject.toml poetry.lock ./
COPY README.md ./
RUN --mount=type=ssh \
    poetry install --no-root --without dev && \
    rm -rf $POETRY_CACHE_DIR

ENTRYPOINT ["poetry", "shell"]
