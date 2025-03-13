##conversation.py
from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
from PIL.Image import Image
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()
    INTERPRETER = auto()
    OBSERVATION = auto()

    def __str__(self):
        if self == Role.SYSTEM:
            return "<|system|>"
        elif self == Role.USER:
            return "<|user|>"
        elif self in [Role.ASSISTANT, Role.TOOL, Role.INTERPRETER]:
            return "<|assistant|>"
        elif self == Role.OBSERVATION:
            return "<|observation|>"
        else:
            raise ValueError(f'Unexpected role: {self}')


    # Get the message block for the given role
    def get_message(self):
        # Compare by value here, because the enum object in the session state
        # is not the same as the enum cases here, due to streamlit's rerunning
        # behavior.
        if self.value == Role.SYSTEM.value:
            return
        elif self.value == Role.USER.value:
            return st.chat_message(name="user", avatar="user")
        elif self.value == Role.ASSISTANT.value:
            return st.chat_message(name="assistant", avatar="assistant")
        elif self.value == Role.TOOL.value:
            return st.chat_message(name="tool", avatar="assistant")
        elif self.value == Role.INTERPRETER.value:
            return st.chat_message(name="interpreter", avatar="assistant")
        elif self.value == Role.OBSERVATION.value:
            return st.chat_message(name="observation", avatar="user")
        else:
            st.error(f'Unexpected role: {self}')


@dataclass
class Conversation:
    role: Role
    content: str
    tool: str | None = None
    image: Image | None = None

    def __str__(self) -> str:
        print(self.role, f"{self.content}", self.tool)
        if self.role in [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.OBSERVATION]:
            return f'{self.role}\n{self.content}'
        elif self.role == Role.TOOL:
            return f'{self.role}{self.tool}\n{self.content}'
        elif self.role == Role.INTERPRETER:
            return f'{self.role}interpreter\n{self.content}'

    # Human readable format
    def get_text(self) -> str:
        # text = postprocess_text(self.content)
        text = self.content  # 只考虑文本情况
        if self.role.value == Role.TOOL.value:
            text = f'Calling tool `{self.tool}`:\n\n{text}'
        elif self.role.value == Role.INTERPRETER.value:
            text = f'{text}'
        elif self.role.value == Role.OBSERVATION.value:
            text = f'Observation:\n```\n{text}\n```'
        return text

    # Display as a markdown block
    def show(self, placeholder: DeltaGenerator | None = None) -> str:
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()
        # if self.image:
        #     message.image(self.image)
        # else:
        #     text = self.get_text()
        #     message.markdown(text)
        if isinstance(self.content, (list, tuple)):
            for content in self.content:
                message.markdown(content)
        else:
            message.write(self.content)

    def to_dict(self):
        convers = []
        if isinstance(self.content, (list, tuple)):
            for c in self.content:
                convers.append({"role": f"{self.role}", "content": f"{c}"})
        else:
            convers.append({"role": f"{self.role}", "content": f"{self.content}"})
        return convers
