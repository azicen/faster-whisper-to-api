import io
import os
import logging
import typing
import ffmpeg
import subprocess
import asyncio
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, BinaryIO
from faster_whisper import WhisperModel
from pydantic import BaseModel


logger = logging.getLogger("hypercorn")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(
    model="large-v3", device="cuda", compute_type="int8_float16"
) -> WhisperModel:
    """
    读取模型
    """
    # device="cuda", "cpu"
    # compute_type=(GPU with FP16 "float16"), (GPU with INT8 "int8_float16"), (CPU with INT8 "int8")
    return WhisperModel(
        model_size_or_path=model, device=device, compute_type=compute_type
    )


default_model = "large-v3"

model_list: Dict[str, WhisperModel] = {}
model_list[default_model] = load_model(model=default_model)


class ErrorResponse(JSONResponse):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        headers: typing.Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTasks | None = None,
    ) -> None:
        super().__init__(
            content={"message": message},
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )


class Word(BaseModel):
    start: float
    end: float
    word: str


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscriptionResponse(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: Optional[List[Segment]] = None
    words: Optional[List[str]] = None


def is_wav_file(file: BinaryIO) -> bool:
    """
    判断文件是否为wav
    """
    try:
        # 读取前12个字节
        header = file.read(12)
        # WAV文件以RIFF标识符开始，后面跟随WAVE类型
        result = header[0:4] == b"RIFF" and header[8:12] == b"WAVE"
        return result
    except (IOError, AttributeError):
        # 如果文件无法打开或读取，返回False
        return False


async def convert_to_wav(file: BinaryIO, file_type: str) -> BinaryIO:
    """
    将目标文件转换为wav
    """
    # 使用ffmpeg-python库转换文件
    cmd = (
        ffmpeg.input("pipe:0", format=file_type)
        .output("pipe:1", format="wav")
        .overwrite_output()
        .compile()
    )

    # 使用subprocess启动ffmpeg进程
    process = await asyncio.create_subprocess_exec(
        *cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # 异步发送输入数据并关闭stdin以表明输入结束
    input_bytes = file.read()
    stdout, _ = await process.communicate(input=input_bytes)

    if process.returncode != 0:
        raise Exception(
            f"FFmpeg process returned error (return code {process.returncode})"
        )

    output_data = io.BytesIO(stdout)

    del stdout, input_bytes
    return output_data


@app.post("/v1/chat/completions")
async def chat_completions():
    """
    `/v1/chat/completions` 接口的模拟实现, 用于oneapi的ping测试, 返回状态码 200 和空字符串
    """
    return PlainTextResponse(content="", status_code=200)


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default_model),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: List[str] = Form(default=[]),
):
    """
    `/v1/audio/transcriptions` 接口的模拟实现

    参数详细说明参见 OpenAI 文档: https://platform.openai.com/docs/api-reference/audio/createTranscription
    """
    task = "transcribe"

    file.file.fileno()

    # 将openai的whisper-1替换为默认模型
    if model == "whisper-1":
        model = default_model

    # 模型是否存在
    if not model in model_list:
        return ErrorResponse(message=f"无法使用不存在的模型: {model}")

    # 获取文件扩展名
    _, file_extension = os.path.splitext(file.filename)
    file_type = file_extension[1:]

    audio_file = file.file
    # 确保文件指针在开始位置
    audio_file.seek(0)
    if not is_wav_file(audio_file):
        audio_file.seek(0)
        # 将其他格式音频转换为wav
        tmp_file = await convert_to_wav(audio_file, file_type)
        audio_file.close()
        audio_file = tmp_file

    segments = []
    try:
        audio_file.seek(0)
        # 使用 faster_whisper 进行转写
        faster_segments, info = model_list[model].transcribe(
            audio=audio_file,
            beam_size=5,
            language=language,
            temperature=temperature,
            initial_prompt=prompt,
            word_timestamps=("word" in timestamp_granularities),
            task=task,
            condition_on_previous_text=False,
        )
        for segment in faster_segments:
            segments.append(segment)
    except Exception as e:
        logger.error(f"模型处理过程中出现错误: {e}")
        # 重新读取模型
        model_list[model] = load_model()
        return ErrorResponse(message="模型处理过程中出现错误")
    finally:
        audio_file.close()

    texts: List[str] = [segment.text for segment in segments]

    transcription_response = TranscriptionResponse(
        task=task, language=info.language, duration=info.duration, text="".join(texts)
    )

    if "segment" in timestamp_granularities:
        transcription_response.segments = [
            Segment(
                id=segment.id,
                seek=segment.seek,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=segment.tokens,
                temperature=segment.temperature,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
            )
            for segment in segments
        ]

    if "word" in timestamp_granularities:
        transcription_response.words = [
            Word(start=word.start, end=word.end, word=word.word)
            for segment in segments
            for word in (segment.words if segment.words is not None else [])
        ]

    if response_format == "json":
        return JSONResponse(
            content=jsonable_encoder(transcription_response), status_code=200
        )
    else:
        # TODO: 根据response_format参数生成不同格式的响应
        return PlainTextResponse(content="", status_code=200)
