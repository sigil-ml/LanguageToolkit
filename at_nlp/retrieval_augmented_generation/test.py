import requests
import fire


def main(prompt: str):
    url = f"http://0.0.0.0:5000/rag_generate?prompt={prompt}"
    resp = requests.get(url)
    print(resp.text)


if __name__ == "__main__":
    fire.Fire(main)
