import docker
import argparse
from loguru import logger as log
from pathlib import Path

client = docker.from_env()

__IMAGE_LIST__ = ["llm-server:latest", "api:latest"]


def build_image(image_name: str, dockerfile: str, tag: str) -> None:
    assert image_name is not None, "image_name cannot be None!"
    assert dockerfile is not None, "dockerfile cannot be None!"
    assert tag is not None, "tag cannot be None!"
    try:
        log.info(f"Building {image_name} image...")
        _, log_gen = client.images.build(
            path="",
            dockerfile=dockerfile,
            tag=tag,
            quiet=False,
            nocache=False,
            rm=True,
        )
        log.info(f"{image_name} built!")
        print("========================================")
        print("BUILD LOGS:")
        # for line in log_gen:
        #     if line:
        #         line_str = line["stream"]
        #         if line_str != "\n":
        #             print(line_str)
        print("========================================")
    except docker.errors.BuildError as e:
        log.critical(f"Build error: {e}")
    except docker.errors.APIError as e:
        log.critical(f"API error: {e}")
    except TypeError as e:
        log.critical(f"Type error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script builds the containers for the LLM project.",
        epilog="By Athena",
    )

    parser.add_argument(
        "-f", "--force", type=bool, help="Force container rebuild", default=False
    )
    args = parser.parse_args()

    image_list = client.images.list()
    image_tags = [image.tags[0] for image in image_list if len(image.tags) > 0]
    for image_name in __IMAGE_LIST__:
        if image_name in image_tags:
            log.info(f"Found {image_name} image!")
            if args.force:
                log.info(f"Rebuilding {image_name} image...")
                if image_name == "llm-server:latest":
                    build_image(
                        "llm-server",
                        "./build/llm-server/llm-server.Dockerfile",
                        "llm-server:latest",
                    )
                elif image_name == "api:latest":
                    build_image("api", "./build/api/api.Dockerfile", "api:latest")
                else:
                    log.critical(f"Unknown image name: {image_name}!")
                    exit()
        else:
            log.info(f"Building {image_name} image...")
            if image_name == "llm-server:latest":
                build_image(
                    "llm-server",
                    "./build/llm-server/llm-server.Dockerfile",
                    "llm-server:latest",
                )
            elif image_name == "api:latest":
                build_image("api", "./build/api/api.Dockerfile", "api:latest")
            else:
                log.critical(f"Unknown image name: {image_name}!")
                exit()
