"""Send one request to the OpenAI-compatible model server."""

import argparse
import json
import os
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        default=os.getenv("MODEL_API_BASE", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "retail-agent"))
    parser.add_argument("--prompt", default="我要黑色M码男士防晒衣，预算300元。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是谨慎的电商客服。信息不足时先追问；"
                    "不要编造库存、订单状态或平台政策。"
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        f"{args.api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Model request failed: {exc}") from exc

    print(body["choices"][0]["message"]["content"])
    print(json.dumps(body.get("usage", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
