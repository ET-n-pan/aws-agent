FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install uv (for uvx command)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Install Go for ARM64
RUN wget https://go.dev/dl/go1.24.10.linux-arm64.tar.gz \
    && tar -C /usr/local -xzf go1.24.10.linux-arm64.tar.gz \
    && rm go1.24.10.linux-arm64.tar.gz

# Set Go environment
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/root/go"
ENV PATH="${GOPATH}/bin:${PATH}"

# Install awsdac and awsdac-mcp-server
RUN go install github.com/awslabs/diagram-as-code/cmd/awsdac@latest \
    && go install github.com/awslabs/diagram-as-code/cmd/awsdac-mcp-server@latest

# Verify installations
RUN which uv && uv --version \
    && which go && go version \
    && which awsdac && awsdac --version \
    && which awsdac-mcp-server \
    && which dot && dot -V

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]