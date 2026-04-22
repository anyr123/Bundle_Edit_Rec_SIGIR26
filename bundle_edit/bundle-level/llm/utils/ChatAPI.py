'''
Description: 
Author: anyiran
Date: 2024-10-14 15:32:07
LastEditors: anyiran
LastEditTime: 2024-11-18 21:20:33
'''
import backoff
import openai
import requests
import json


class OpenAI:
    def __init__(self, model, api_key, temperature=0):


        openai.api_key = ''
        openai.api_base=''
        self.model = model
        self.temperature = temperature

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout), max_tries=10, factor=2, max_time=120)
    def create_chat_completion(self, messages):

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content

class Claude:
    def __init__(self, model, api_key, temperature=0):

        openai.api_key = ''
        openai.api_base=''
        self.model = model
        self.temperature = temperature

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout), max_tries=10, factor=2, max_time=120)
    def create_chat_completion(self, messages):

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content

