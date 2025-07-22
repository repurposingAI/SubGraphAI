from groq import Groq
from graph_maker.o_types import LLMClient

client = Groq(api_key="")


class GroqClient(LLMClient):
    _model: str
    _temperature: float
    _top_p: float


    def __init__(self, model: str = "deepseek-r1-distill-llama-70b", temperature=0.6, top_p=1):

        self._model = model
        self._temperature = temperature
        self._top_p = top_p

    def generate(self, user_message: str, system_message: str) -> str:
        print("Using Model: ", self._model)
        result = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            
            model=self._model,
           
            temperature=self._temperature,
           
            top_p=self._top_p,
           
            stop=None,
         
            stream=False,
        )

        return result.choices[0].message.content
