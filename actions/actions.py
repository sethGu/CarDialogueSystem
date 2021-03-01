# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"


from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import json
from typing import Text, Dict, Any, List, Union

from rasa_sdk.events import SlotSet
from rasa_sdk import Action, Tracker
import re
from schema import schema
from graph_database import GraphDatabase


# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
#
# class ActionSearchcar(Action):
#     def name(self) -> Text:
#         return "action_search_car_info"
#
#     def run(
#             self,
#             dispatcher,
#             tracker: Tracker,
#             domain: "DomainDict",
#     ) -> List[Dict[Text, Any]]:
#         car_type = tracker.get_slot("car_type")
#         car_part = tracker.get_slot("car_part")
#
#         answer = search_car_info(car_type,car_part)
#         if answer == 'None':
#             dispatcher.utter_message(template="utter_wrong_answer")
#         else:
#             dispatcher.utter_message(template="utter_answer",answer = answer,car_type = car_type, car_part = car_part)
#         return []
#
# def search_car_info(car_type,car_part):
#     synonym=['2021款 2.4L汽油手动两驱精英版长厢高底盘','新款汽车','2021款两驱','2021款高底盘','大通2021款新车','今年大通新款汽车','2021新款','2021']
#     if car_type in synonym:
#         car_type=synonym[0]
#     print(car_type)
#     print(car_part)
#     knowledge_graph = Graph('http://127.0.0.1:7474', username='neo4j', password='123456')
#     query_cql="match  (n:车型)-[:配置]->(p:基本参数) where n.name='"+car_type+"' return p."+car_part
#     p=knowledge_graph.run(query_cql)
#     try:
#         return str(list(p)[0])
#     except:
#         return 'None'

def date_transfer(date):
    """
    将上市时间标准化
    :param date:
    :return:
    """
    if date is None:
        return None
    else:
        number = re.findall(r'\d+', date)
        if len(number) == 1 and len(number[0]) == 4:
            return number[0]
        elif len(number) == 1 and len(number[0]) == 2:
            return "20" + number[0]
        elif len(number) == 2:
            return None
        else:
            return None




class ActionQuerycarseries(Action):
    """
    查询车系
    """

    def name(self):
        return "action_query_car_series"

    def run(self, dispatcher, tracker, domain):
        graph_database = GraphDatabase()
        car_series_list = graph_database.get_entities(entity_type='车系')
        dispatcher.utter_message(template="utter_answer", answer="小通找到了下列车系：")
        car_series_list_slot = ['车系']
        for i, e in enumerate(car_series_list):
            answer = str(i + 1) + ": " + e['name']
            dispatcher.utter_message(template="utter_answer", answer=answer)
            car_series_list_slot.append(e['name'])
        slots = [
            SlotSet("listed_items", car_series_list_slot)
        ]
        return slots


class ActionQuerycarseries2carmodel(Action):
    """
    查某车系的车型
    """
    def name(self):
        return "action_query_car_series_to_car_model"

    def run(self, dispatcher, tracker, domain):
        car_series = tracker.get_slot('car_series')
        if car_series is None:
            dispatcher.utter_message(template="utter_not_clear")
        graph_database = GraphDatabase()
        car_model_list = graph_database.query_relation2entity(entity=car_series, relation='属于')
        dispatcher.utter_message(template="utter_answer",
                                 answer="小通一共找到" + str(len(car_model_list)) + "个车型属于该车系，由于篇幅限制，为您列出以下几种：")
        car_model_list_slot = ['车型']
        for i, e in enumerate(car_model_list):
            if i >= 10:
                break
            answer = str(i + 1) + ": " + e
            dispatcher.utter_message(template="utter_answer", answer=answer)
            car_model_list_slot.append(e)
        slots = [
            SlotSet("listed_items", car_model_list_slot)
        ]
        return slots


class ActionQueryattribute2carmodel(Action):
    """
    查满足属性的车型
    """

    def name(self):
        return "action_query_car_model_by_attribute"

    def run(self, dispatcher, tracker, domain):
        car_body = tracker.get_slot('car_body')
        energy_type = tracker.get_slot('energy_type')
        time2market = tracker.get_slot('time2market')
        listed_items = tracker.get_slot('listed_items')
        SlotSet("car_body", None)
        SlotSet("energy_type", None)
        SlotSet("time2market", None)
        attribute_dict = {}
        attribute_dict['body_structure'] = car_body
        attribute_dict['energy_type'] = energy_type
        if time2market:
            attribute_dict['time2market'] = date_transfer(time2market)
        else:
            attribute_dict['time2market'] = None
        graph_database = GraphDatabase()
        car_model_list = graph_database.query_attribute2entity(attribute_dict)
        dispatcher.utter_message(template="utter_answer",
                                 answer="小通一共找到" + str(len(car_model_list)) + "个车型，由于篇幅限制，为您列出以下几种：")
        car_model_list_slot = []
        for i, e in enumerate(car_model_list):
            if i >= 10:
                break
            answer = str(i + 1) + ": " + e
            dispatcher.utter_message(template="utter_answer", answer=answer)
            car_model_list_slot.append(e)
        slots = [
            SlotSet("listed_items", car_model_list_slot)
        ]
        return slots


class ActionQuerycarmodel2attribute(Action):
    """
    查某车型的某个属性
    """

    def name(self):
        return "action_query_attribute_in_car_model"

    def run(self, dispatcher, tracker, domain):
        car_series = tracker.get_slot('car_series')
        attribute = tracker.get_slot('attribute')
        relationship = tracker.get_slot('relationship')
        SlotSet("attribute", None)
        SlotSet("relationship", None)
        if car_series is None:
            dispatcher.utter_message(template="utter_not_clear")
            return []
        elif attribute is None and relationship is None:
            dispatcher.utter_message(template="utter_not_clear")
            return []
        elif attribute:
            graph_database = GraphDatabase()
            car_model_dict = graph_database.query_entity2attribute(entity=car_series, relationship=None,c_attribute=attribute)
            answer="小通为你查询到"+car_series+"的"
            for key in car_model_dict.keys():
                answer += key+"是"+car_model_dict[key]+" "
            dispatcher.utter_message(template="utter_answer",
                                     answer=answer)
            return [SlotSet("car_series",car_series)]
        elif relationship and attribute is None :
            graph_database = GraphDatabase()
            car_model_dict = graph_database.query_entity2attribute(entity=car_series, relationship=relationship,c_attribute=None)
            answer="小通为你查询到"+car_series+"的 "
            for key in car_model_dict.keys():
                answer += key+": "+str(car_model_dict[key])+" "
            dispatcher.utter_message(template="utter_answer",
                                     answer=answer)
            return [SlotSet("car_series",car_series)]

class Actionresolve_entity(Action):
    """
    顺序指代问题
    """

    def name(self):
        return "action_resolve_entity"

    def run(self, dispatcher, tracker, domain):
        listed_items = tracker.get_slot('listed_items')
        if len(listed_items) == 0:
            dispatcher.utter_message(template="utter_no_list")
        elif len(listed_items)>0:
            mention = tracker.get_slot("mention")
            if mention is None:
                dispatcher.utter_message(template="utter_out_of_scope")
            else:
                try:
                    index_of_list = int(mention)
                    answer="已为您锁定查询"+listed_items[0]+" "+listed_items[index_of_list]+",请问您具体想查询什么？"
                    dispatcher.utter_message(template="utter_answer",
                                             answer=answer)
                except:
                    dispatcher.utter_message(template="utter_not_clear")
                    return []




if __name__ == "__main__":
    print(date_transfer('21年'))
