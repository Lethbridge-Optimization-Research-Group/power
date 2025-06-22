# Base Julia image
FROM julia:1.10.3

# Install additional tools (optional)
RUN apt-get update && apt-get install -y git wget curl

# Pre-install Julia packages (optional)
# RUN julia -e 'using Pkg; Pkg.add(["Plots", "JuMP", "DataFrames"])'

# Set default work directory
WORKDIR /workspace

# Copy your project files into the container
COPY . /workspace
