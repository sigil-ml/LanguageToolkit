from loguru import logger as log
from pathlib import Path

import toml
import docker

client = docker.from_env()

if __name__ == "__main__":
    assert Path("./config.toml").exists(), "config.toml not found!"
    conf = toml.load("./config.toml")["llm-server"]
    port = conf["port"]

    model = Path("./models/mixtral8x7b-q4/mixtral-8x7b-v0.1.Q4_K_M.gguf").absolute()
    assert model.exists(), "Model not found!"

    image_list = [img.tags[0] for img in client.images.list()]
    print(image_list)
    assert "llm-server:latest" in image_list, "llm-server image not found!"
    log.info("Starting llm-server container...")
    client.containers.run(
        "llm-server:latest",
        detach=True,
        ports={f"{port}/tcp": port},
        volumes={
            str(model): {
                "bind": "/app/models/mixtral-8x7b-v0.1.Q4_K_M.gguf",
                "mode": "ro",
            }
        },
    )
