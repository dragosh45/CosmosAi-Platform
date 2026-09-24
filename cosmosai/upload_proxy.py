"""Bounded image forwarding shared by the gateway and inference router."""

import requests
from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png"}


def response_json(response: requests.Response) -> dict:
    """Preserve actionable downstream errors without passing HTML error pages."""
    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(502, "The prediction service returned an invalid response") from error
    if not isinstance(payload, dict):
        raise HTTPException(502, "The prediction service returned an invalid response")
    if not 200 <= response.status_code < 300:
        detail = payload.get("detail")
        raise HTTPException(
            response.status_code if 400 <= response.status_code < 600 else 502,
            detail if isinstance(detail, str) else "The prediction service could not complete the request",
        )
    return payload


def upstream_request(method: str, url: str, *, timeout: int = 60, **kwargs) -> dict:
    """Call a configured service; uploaded data never controls the destination."""
    try:
        with requests.request(method, url, timeout=(3, timeout), **kwargs) as response:
            return response_json(response)
    except requests.Timeout as error:
        raise HTTPException(504, "Prediction timed out. Please try again") from error
    except requests.RequestException as error:
        raise HTTPException(502, "The prediction service is unavailable") from error


async def forward_image(request: Request, url: str, *, timeout: int = 60) -> dict:
    """Enforce the byte limit at each hop, including chunked request bodies."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in IMAGE_TYPES:
        raise HTTPException(415, "Choose a JPEG or PNG image")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 5 MiB upload limit")
        content.extend(chunk)
    if not content:
        raise HTTPException(422, "The image is empty")
    return await run_in_threadpool(
        upstream_request, "POST", url, timeout=timeout,
        data=bytes(content), headers={"Content-Type": content_type},
    )
