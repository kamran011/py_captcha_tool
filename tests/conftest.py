"""Tests never read the developer's .env: a real vision key there must not reach a paid API."""
import os

os.environ["CAPTCHA_ENV_FILE"] = os.devnull
for _name in list(os.environ):
    if _name.upper().startswith("VISION_"):
        del os.environ[_name]
