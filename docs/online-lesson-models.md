# Online Lesson AI model deployment

Online Lesson chat and practice question generation are configured in
`gradenza-api` through Pydantic settings. Production can be upgraded or rolled
back by changing environment variables, without editing route code.

Set these backend environment variables to use GPT-5.5:

```env
ONLINE_LESSON_CHAT_MODEL=openai/gpt-5.5
ONLINE_LESSON_QUESTIONS_MODEL=openai/gpt-5.5
```

Rollback values:

```env
ONLINE_LESSON_CHAT_MODEL=google/gemini-2.5-flash
ONLINE_LESSON_QUESTIONS_MODEL=google/gemini-2.5-flash
```

`gradenza-api/src/gradenza_api/settings.py` reads these values through
Pydantic settings. The frontend only proxies Online Lesson AI requests and does
not choose the model. Redeploy or restart `gradenza-api` after backend model env
changes. Redeploy or restart the frontend too if proxy/admin environment values
changed, such as `NEXT_PUBLIC_API_URL` or `INTERNAL_API_SECRET`.

Use the admin diagnostics route or `ai_usage_events.model` rows for
`online_lesson_chat` and `online_lesson_generate_questions` to confirm the
configured and provider-returned model names in production.
