from model.models import record_speech
from Speech_and_Text import text_to_speech
from rasa.core.agent import Agent
from rasa.core.interpreter import RasaNLUInterpreter
import asyncio


# record = record_speech()
# record.record()
# record.recognition()

def interaction(a: str, time: int):
    agent, record = Agent.load(a), record_speech(time=time)
    text_to_speech("您好！这里是对话机器人一期！")
    text_to_speech("请讲！")
    record.record()
    record_text = record.recognition()
    response = asyncio.run(agent.handle_text(record_text))
    print("")
    print("识别结果:")
    print(record_text)
    print(response[0]["text"])
    # print("聊天机器人回复：")
    while len(record_text) > 1:

        while not response:
            print("")
            print("未能识别您的问题，请减少周围杂音后再说一遍")
            text_to_speech(sentence="未能识别您的问题，请减少周围杂音后再说一遍")
            record.record()
            record_text = record.recognition()
            response = asyncio.run(agent.handle_text(record_text))
            print("识别结果:")
            print(record_text)

        while response:
            print("")
            print("机器人的回答是:")
            print(response[0]["text"])
            text_to_speech(sentence=response[0]["text"])
            record.record()
            record_text = record.recognition()
            response = asyncio.run(agent.handle_text(record_text))
            print("")
            print("识别结果:")
            print(record_text)

    print("对话已结束")
    text_to_speech(sentence="对话已结束")


interaction(a="models/20210301-170024.tar.gz", time=5)
