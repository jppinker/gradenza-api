"""
Tutor video generation via OpenRouter.

The route mirrors the legacy Supabase Edge Function used by the old project:
models, prompt enhancement, generation, polling, and download. Completed videos
are copied to Supabase Storage and served from there.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.supabase_client import get_service_client, run_sync
from gradenza_api.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/video", tags=["video"])

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REFERER = "https://www.gradenza.com"
TITLE = "Gradenza Tutor Video"
class VideoActionRequest(BaseModel):
    action: Literal["models", "enhance_prompt", "estimate", "generate", "poll", "download"]
    user_prompt: str | None = Field(default=None, max_length=6000)
    current_prompt: str | None = Field(default=None, max_length=12000)
    command: str | None = Field(default=None, max_length=3000)
    video_model: str | None = Field(default=None, max_length=200)
    parameters: dict[str, Any] | None = None
    job_id: str | None = Field(default=None, max_length=200)
    polling_url: str | None = Field(default=None, max_length=500)
    index: int = Field(default=0, ge=0, le=20)


def _clean(value: Any, max_len: int = 12000) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()[:max_len]


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _parse_openrouter_error(data: Any) -> str:
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        if isinstance(err, str):
            return err
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(data or "")


async def _openrouter_fetch(path_or_url: str, *, method: str = "GET", json_body: Any = None, timeout: float = 120.0) -> Any:
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured")

    url = path_or_url if path_or_url.startswith("http") else f"{OPENROUTER_BASE_URL}{path_or_url}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.request(
            method,
            url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": REFERER,
                "X-Title": TITLE,
            },
            json=json_body,
        )

    data = res.json() if res.headers.get("content-type", "").lower().startswith("application/json") else res.text
    if not res.is_success:
        raise HTTPException(status_code=502, detail=f"OpenRouter {res.status_code}: {_parse_openrouter_error(data)}")
    return data


async def _fetch_video_content(job_id: str, index: int) -> tuple[bytes, str]:
    url = f"{OPENROUTER_BASE_URL}/videos/{job_id}/content?index={index}"
    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Accept": "video/*,application/octet-stream,*/*",
                "HTTP-Referer": REFERER,
                "X-Title": TITLE,
            },
        )

    if not res.is_success:
        details = res.text[:1000]
        raise HTTPException(status_code=502, detail=f"OpenRouter video content request failed: {res.status_code} {details}")

    content_type = res.headers.get("content-type") or "video/mp4"
    return res.content, content_type


async def _fetch_video_models() -> list[dict[str, Any]]:
    data = await _openrouter_fetch("/videos/models")
    return data.get("data", []) if isinstance(data, dict) and isinstance(data.get("data"), list) else []


def _pick_size(model: dict[str, Any] | None, resolution: str | None, aspect_ratio: str | None) -> dict[str, Any] | None:
    sizes = model.get("supported_sizes") if isinstance(model, dict) else None
    if not isinstance(sizes, list) or not sizes:
        return None
    try:
        rw, rh = [float(part) for part in (aspect_ratio or "").split(":", 1)]
    except ValueError:
        rw, rh = 0, 0
    target_height = {"4K": 2160, "1080p": 1080, "720p": 720, "480p": 480}.get(resolution or "")
    target_width = round((target_height * rw) / rh) if target_height and rw and rh else None

    parsed: list[tuple[str, int, int]] = []
    for size in sizes:
        if not isinstance(size, str) or "x" not in size:
            continue
        try:
            width, height = [int(part) for part in size.split("x", 1)]
        except ValueError:
            continue
        parsed.append((size, width, height))

    if not parsed:
        return None

    chosen = next((item for item in parsed if item[1] == target_width and item[2] == target_height), None)
    chosen = chosen or next((item for item in parsed if item[2] == target_height), None) or parsed[0]
    return {"size": chosen[0], "width": chosen[1], "height": chosen[2]}


def _estimate_cost(model: dict[str, Any] | None, params: dict[str, Any]) -> dict[str, Any]:
    pricing = model.get("pricing_skus") if isinstance(model, dict) else None
    pricing = pricing if isinstance(pricing, dict) else {}
    durations = model.get("supported_durations") if isinstance(model, dict) else None
    resolutions = model.get("supported_resolutions") if isinstance(model, dict) else None
    aspect_ratios = model.get("supported_aspect_ratios") if isinstance(model, dict) else None
    duration = _safe_number(params.get("duration")) or (durations[0] if isinstance(durations, list) and durations else 4)
    resolution = _clean(params.get("resolution"), 20) or (resolutions[0] if isinstance(resolutions, list) and resolutions else "720p")
    aspect_ratio = _clean(params.get("aspect_ratio"), 20) or (aspect_ratios[0] if isinstance(aspect_ratios, list) and aspect_ratios else "16:9")
    with_audio = params.get("generate_audio") is not False and (not isinstance(model, dict) or model.get("generate_audio") is not False)
    size = _pick_size(model, resolution, aspect_ratio)

    if pricing.get("video_tokens") or pricing.get("video_tokens_without_audio"):
        raw = pricing.get("video_tokens" if with_audio else "video_tokens_without_audio") or pricing.get("video_tokens") or pricing.get("video_tokens_without_audio")
        price_per_token = _safe_number(raw)
        if size and price_per_token is not None:
            tokens = (size["width"] * size["height"] * duration * 24) / 1024
            return {"cost": tokens * price_per_token, "confidence": "calculated", "formula": f'{size["size"]}, {duration:g}s, 24 fps token estimate'}

    resolution_key = resolution.lower()
    candidates = [
        "duration_seconds_with_audio_4k" if with_audio and resolution == "4K" else "",
        "duration_seconds_without_audio_4k" if not with_audio and resolution == "4K" else "",
        f"duration_seconds_with_audio_{resolution_key}" if with_audio else "",
        f"duration_seconds_without_audio_{resolution_key}" if not with_audio else "",
        f"duration_seconds_{resolution_key}",
        "duration_seconds_with_audio" if with_audio else "",
        "duration_seconds_without_audio" if not with_audio else "",
        f"text_to_video_duration_seconds_{resolution_key}",
        "duration_seconds",
    ]
    for key in [item for item in candidates if item]:
        value = _safe_number(pricing.get(key))
        if value is not None:
            return {"cost": value * duration, "confidence": "calculated", "formula": f"{key} x {duration:g}s"}

    cents = _safe_number(pricing.get(f"cents_per_video_output_second_{resolution_key}"))
    if cents is not None:
        return {"cost": (cents / 100) * duration, "confidence": "calculated", "formula": f"{cents:g} cents/s x {duration:g}s"}

    return {"cost": None, "confidence": "unknown", "formula": "OpenRouter did not expose a matching price SKU."}


def _build_video_payload(params: dict[str, Any]) -> dict[str, Any]:
    model = _clean(params.get("model"), 200)
    negative_prompt = _clean(params.get("negative_prompt"), 2000)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": _clean(params.get("prompt"), 12000),
    }
    for key in ("duration", "resolution", "aspect_ratio"):
        if params.get(key):
            payload[key] = params[key]
    if isinstance(params.get("generate_audio"), bool):
        payload["generate_audio"] = params["generate_audio"]
    seed = _safe_number(params.get("seed"))
    if seed is not None:
        payload["seed"] = int(seed)

    if model.startswith("google/"):
        google_params: dict[str, Any] = {}
        if negative_prompt:
            google_params["negativePrompt"] = negative_prompt
        if isinstance(params.get("enhance_prompt"), bool):
            google_params["enhancePrompt"] = params["enhance_prompt"]
        if params.get("person_generation") in {"allow", "dont_allow", "allow_adult"}:
            google_params["personGeneration"] = params["person_generation"]
        if google_params:
            payload["provider"] = {"options": {"google-vertex": {"parameters": google_params}}}
    elif negative_prompt:
        payload["negative_prompt"] = negative_prompt

    cfg_scale = _safe_number(params.get("cfg_scale"))
    if model.startswith("kwaivgi/") and cfg_scale is not None:
        payload["cfg_scale"] = cfg_scale
    return payload


def _strip_fences(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


async def _enhance_prompt(body: VideoActionRequest) -> dict[str, Any]:
    user_prompt = _clean(body.user_prompt, 6000)
    current_prompt = _clean(body.current_prompt, 12000)
    command = _clean(body.command, 3000)
    model_id = _clean(body.video_model, 200)
    parameters = body.parameters or {}

    if not user_prompt and not current_prompt:
        raise HTTPException(status_code=422, detail="Prompt text is required")

    instruction = (
        "Revise the existing video-generation prompt according to the user's command."
        if command
        else "Transform the user's short idea into a production-ready video-generation prompt."
    )
    messages = [
        {
            "role": "system",
            "content": " ".join(
                [
                    "You are a senior AI video prompt director.",
                    "Write only the final video prompt, no markdown and no commentary.",
                    "Make the result specific: subject, action, setting, shot sequence, camera movement, lighting, style, motion timing, audio notes when relevant, and avoid ambiguous words.",
                    "Keep it practical for text-to-video models. Do not add copyrighted characters, living public figures, brand logos, or unsafe content unless the user explicitly provided rights and safety context.",
                ]
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                item
                for item in [
                    instruction,
                    f"Target video model: {model_id or 'not selected'}",
                    f"Selected parameters: {parameters}",
                    f"Existing prompt:\n{current_prompt}" if current_prompt else "",
                    f"User revision command:\n{command}" if command else f"User idea:\n{user_prompt}",
                ]
                if item
            ),
        },
    ]

    async def request(model: str) -> dict[str, Any]:
        data = await _openrouter_fetch(
            "/chat/completions",
            method="POST",
            json_body={"model": model, "messages": messages, "temperature": 0.45, "max_tokens": 1800, "usage": {"include": True}},
        )
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="OpenRouter returned an invalid prompt response")
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=502, detail="OpenRouter returned an empty prompt")
        return data

    used_model = settings.openrouter_video_prompt_model
    try:
        data = await request(used_model)
    except Exception:
        fallback = settings.openrouter_video_prompt_fallback_model
        if not fallback or fallback == used_model:
            raise
        used_model = fallback
        data = await request(used_model)

    content = data["choices"][0]["message"]["content"]
    return {"prompt": _strip_fences(content), "model": data.get("model") or used_model, "usage": data.get("usage")}


async def _upload_to_storage(user_id: str, job_id: str, content: bytes, content_type: str) -> tuple[str, str]:
    storage_path = f"{user_id}/video-generations/{job_id}.mp4"
    url = f"{settings.supabase_url}/storage/v1/object/{settings.tutor_video_bucket}/{storage_path}"
    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.put(
            url,
            headers={
                "Authorization": f"Bearer {settings.supabase_service_role_key}",
                "apikey": settings.supabase_service_role_key,
                "Content-Type": content_type or "video/mp4",
                "x-upsert": "true",
            },
            content=content,
        )
    if not res.is_success:
        raise HTTPException(status_code=502, detail=f"Supabase storage upload failed: {res.status_code} {res.text[:500]}")
    return settings.tutor_video_bucket, storage_path


async def _download_from_storage(bucket: str, storage_path: str) -> tuple[bytes, str]:
    url = f"{settings.supabase_url}/storage/v1/object/{bucket}/{storage_path}"
    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {settings.supabase_service_role_key}",
                "apikey": settings.supabase_service_role_key,
            },
        )
    if not res.is_success:
        raise HTTPException(status_code=502, detail=f"Supabase storage download failed: {res.status_code} {res.text[:500]}")
    return res.content, res.headers.get("content-type") or "video/mp4"


async def _save_generation(row: dict[str, Any], *, on_conflict: str = "job_id") -> None:
    svc = get_service_client()

    def _upsert() -> None:
        svc.table("tutor_video_generations").upsert(row, on_conflict=on_conflict).execute()

    await run_sync(_upsert)


async def _update_generation(job_id: str, row: dict[str, Any]) -> dict[str, Any] | None:
    svc = get_service_client()

    def _update() -> dict[str, Any] | None:
        result = (
            svc.table("tutor_video_generations")
            .update(row)
            .eq("job_id", job_id)
            .execute()
        )
        data = result.data or []
        return data[0] if data else None

    return await run_sync(_update)


async def _get_generation(job_id: str) -> dict[str, Any] | None:
    svc = get_service_client()

    def _fetch() -> dict[str, Any] | None:
        return (
            svc.table("tutor_video_generations")
            .select("*")
            .eq("job_id", job_id)
            .maybe_single()
            .execute()
            .data
        )

    return await run_sync(_fetch)


@router.post("")
async def video_action(
    body: VideoActionRequest,
    user: Annotated[AuthUser, Depends(require_roles("teacher", "tutor"))],
) -> Any:
    if not user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User-scoped auth is required")

    if body.action == "models":
        return {"data": await _fetch_video_models()}

    if body.action == "enhance_prompt":
        return await _enhance_prompt(body)

    if body.action == "estimate":
        params = body.parameters or {}
        models = await _fetch_video_models()
        model = next((item for item in models if item.get("id") == params.get("model")), None)
        return {"estimate": _estimate_cost(model, params), "model": model}

    if body.action == "generate":
        params = body.parameters or {}
        model_id = _clean(params.get("model"), 200)
        prompt = _clean(params.get("prompt"), 12000)
        if not model_id:
            raise HTTPException(status_code=422, detail="model is required")
        if not prompt:
            raise HTTPException(status_code=422, detail="prompt is required")

        models = await _fetch_video_models()
        model = next((item for item in models if item.get("id") == model_id), None)
        if not model:
            raise HTTPException(status_code=422, detail="Unknown video model")

        estimate = _estimate_cost(model, params)
        payload = _build_video_payload(params)
        data = await _openrouter_fetch("/videos", method="POST", json_body=payload, timeout=180.0)
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="OpenRouter returned an invalid generation response")

        job_id = _clean(data.get("id"), 200)
        if job_id:
            await _save_generation(
                {
                    "job_id": job_id,
                    "user_id": user.id,
                    "model": model_id,
                    "prompt": prompt,
                    "parameters": payload,
                    "estimated_cost": estimate.get("cost") if isinstance(estimate.get("cost"), (int, float)) else None,
                    "status": data.get("status") or "pending",
                    "polling_url": data.get("polling_url"),
                }
            )

        return {"job": data, "estimate": estimate}

    if body.action == "poll":
        job_id = _clean(body.job_id, 200)
        polling_url = _clean(body.polling_url, 500)
        if not job_id and not polling_url:
            raise HTTPException(status_code=422, detail="job_id or polling_url is required")

        path = polling_url or f"/videos/{job_id}"
        data = await _openrouter_fetch(path)
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="OpenRouter returned an invalid poll response")

        resolved_id = _clean(data.get("id"), 200) or job_id
        status_value = _clean(data.get("status"), 60)
        actual_cost = data.get("usage", {}).get("cost") if isinstance(data.get("usage"), dict) else None
        unsigned_urls = data.get("unsigned_urls") if isinstance(data.get("unsigned_urls"), list) else []
        video_url = unsigned_urls[0] if unsigned_urls else None

        existing = await _get_generation(resolved_id) if resolved_id else None
        if existing and existing.get("user_id") != user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

        storage_bucket = existing.get("storage_bucket") if existing else None
        storage_path = existing.get("storage_path") if existing else None
        if resolved_id and status_value == "completed" and not storage_path:
            content, content_type = await _fetch_video_content(resolved_id, 0)
            storage_bucket, storage_path = await _upload_to_storage(user.id, resolved_id, content, content_type)

        if resolved_id:
            await _update_generation(
                resolved_id,
                {
                    "status": status_value,
                    "actual_cost": actual_cost if isinstance(actual_cost, (int, float)) else None,
                    "content_available": bool(storage_path or unsigned_urls),
                    "video_url": video_url,
                    "storage_bucket": storage_bucket,
                    "storage_path": storage_path,
                    "error": data.get("error"),
                    "raw_response": data,
                },
            )

        if storage_path:
            data["storage_bucket"] = storage_bucket
            data["storage_path"] = storage_path
            data["stored"] = True
        return {"job": data}

    if body.action == "download":
        job_id = _clean(body.job_id, 200)
        if not job_id:
            raise HTTPException(status_code=422, detail="job_id is required")

        row = await _get_generation(job_id)
        if row and row.get("user_id") != user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        if row and row.get("status") not in (None, "completed"):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Video is not completed yet: {row.get('status')}")

        bucket = row.get("storage_bucket") if row else None
        storage_path = row.get("storage_path") if row else None
        if bucket and storage_path:
            content, content_type = await _download_from_storage(bucket, storage_path)
        else:
            content, content_type = await _fetch_video_content(job_id, max(0, body.index))
            bucket, storage_path = await _upload_to_storage(user.id, job_id, content, content_type)
            await _update_generation(
                job_id,
                {"storage_bucket": bucket, "storage_path": storage_path, "content_available": True},
            )

        headers = {
            "Content-Disposition": f'attachment; filename="gradenza-video-{job_id}.mp4"',
            "Cache-Control": "private, no-store",
        }
        return Response(content=content, media_type=content_type, headers=headers)

    raise HTTPException(status_code=400, detail="Unknown action")
