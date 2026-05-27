# Transcript

Локальный транскрибатор аудио/видео. WhisperX (faster-whisper + pyannote диаризация) + FastAPI + React/TS.

## Стек

- **Backend**: FastAPI + whisperX (faster-whisper, word-align, pyannote diarization), SSE прогресс
- **Frontend**: Vite + React + TypeScript
- **Все в Docker**, ffmpeg внутри

## Запуск

1. Скопировать env:
   ```
   cp .env.example .env
   ```
2. (Опционально) Для диаризации спикеров — заполнить `HF_TOKEN`:
   - Зарегистрироваться huggingface.co (бесплатно)
   - Принять условия: <https://huggingface.co/pyannote/speaker-diarization-3.1> и <https://huggingface.co/pyannote/segmentation-3.0>
   - Создать токен (read) на <https://huggingface.co/settings/tokens>
   - Вставить в `.env` как `HF_TOKEN=hf_...`
3. Старт:
   ```
   docker compose up --build
   ```
4. Открыть <http://localhost:5173>

Первый запуск — долгий (качает модель whisper ~500MB для `small` + pyannote ~500MB). Кэшируется в volume.

## Конфиг

В `docker-compose.yml` → `backend.environment`:
- `WHISPER_MODEL` — `tiny | base | small | medium | large-v3` (рус ок с `small`+)
- `WHISPER_DEVICE` — `cpu` или `cuda`
- `WHISPER_COMPUTE_TYPE` — `int8` (cpu), `float16` (cuda), `float32`

## GPU

Раскомментировать в `docker-compose.yml` (добавить):
```yaml
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
```
И поставить `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`. Базовый образ заменить на `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04`.

## API

- `POST /api/jobs` — multipart: `file`, `diarize` (bool), `language` (optional) → `{id}`
- `GET /api/jobs/{id}/events` — SSE стрим прогресса
- `GET /api/jobs/{id}` — финальный результат
- `GET /api/jobs/{id}/srt` — SRT субтитры

## TODO будущее

- Очередь задач (Redis/RQ) для нескольких параллельных загрузок
- Auth + per-user storage
- Sqlite/Postgres для истории
- Стрим записи с микрофона
