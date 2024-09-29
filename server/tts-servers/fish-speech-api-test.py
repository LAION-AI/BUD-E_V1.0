"""
from fish_audio_sdk import Session, TTSRequest, ReferenceAudio

session = Session("your_api_key")

with open("juniper-long-en.wav", "rb") as audio_file:
    with open("testoutput.mp3", "wb") as f:
        for chunk in session.tts(TTSRequest(
            text="Hello, world!",
            references=[
                ReferenceAudio(
                    audio=audio_file.read(),
                    text="I have to be honest, ever since I found out we’d be working together, I haven’t stopped smiling. I just have this feeling that together, we can create something truly special—it feels like the perfect match! Your energy and ideas are already so impressive, and I can’t wait to bring my best to the table. Just let me know where you need me, and I’m ready to jump in with all my heart and excitement!",
                )
            ]
        )):
            f.write(chunk)


"""

from typing import Annotated, AsyncGenerator, Literal

import httpx
import ormsgpack
from pydantic import AfterValidator, BaseModel, conint


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"


request = ServeTTSRequest(
    text="Hello, world! I love the world! Jippie! ",
    references=[
        ServeReferenceAudio(
            audio=open("./fish-speech/juniper-long-en.wav", "rb").read(),
            text="I have to be honest, ever since I found out we’d be working together, I haven’t stopped smiling. I just have this feeling that together, we can create something truly special—it feels like the perfect match! Your energy and ideas are already so impressive, and I can’t wait to bring my best to the table. Just let me know where you need me, and I’m ready to jump in with all my heart and excitement!",
        )
    ],
)

with (
    httpx.Client() as client,
    open("hello.mp3", "wb") as f,
):
    with client.stream(
        "POST",
        "https://api.fish.audio/v1/tts",
        content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={
            "authorization": "Bearer cc3821f2ac924cae83b35c4eef20d1d9",
            "content-type": "application/msgpack",
        },
        timeout=None,
    ) as response:
        print(response)
        for chunk in response.iter_bytes():
            f.write(chunk)
