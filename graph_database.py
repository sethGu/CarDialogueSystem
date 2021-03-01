from typing import List, Dict, Any, Optional, Text

from py2neo import Graph,Node,data,Path, Relationship,NodeMatcher
from schema import schema,slot_neo4j_dict

class KnowledgeBase(object):
    def get_entities(self,
                     entity_type:Text,
                     attributes:Optional[List[Dict[Text,Text]]] = None
                     ) -> List[Dict[Text,Any]]:
        '''在图形数据库中查询给定类型的实体。如果给定了任何属性，则通过提供的属性限制实体。'''
        raise NotImplementedError("get_entities 函数没有实现.")

    def get_attribute_of(self,
                         entity_type:Text,
                         key_attribute:Text,
                         entity:Text,
                         attribute:Text
                         ) -> List[Any]:
        '''获取所提供实体的给定属性的值'''
        raise NotImplementedError("get_attribute_of 函数没有实现.")

    def validate_entity(self,
                        eneity_type,
                        entity,
                        key_attribute,
                        attributes
                        ) -> Optional[Dict[Text,Any]]:
        '''验证给定实体是否具有所有提供的属性值'''
        raise NotImplementedError("validate_entity 函数没有实现.")

class GraphDatabase(KnowledgeBase):

    def __init__(self,uri= 'http://localhost:7474',
                 username='neo4j', password='123456'):
        self.uri = uri
        self.username = username
        self.password = password
        self.graph = Graph('http://localhost:7474', username='neo4j', password='123456')

    def _node_to_dict(self,node):
        '''将node转变为字典'''
        id = node.identity
        label = str(node.labels).split(':')[1] #node的label有点难取。。。
        entity = {'id':id,'label':label}
        for key,value in node.items():
            entity[key] = value
        return entity

    # def _execute_entity_query(self,query:Text) -> List[Dict[Text,Any]]:
    #     ''''''


    def get_entities(self,
                     entity_type:Text = None,
                     attributes:Optional[List[Dict[Text,Text]]] = None,
                     limit: int =10
                     ):
        if entity_type == None and attributes == None:
            raise Exception('GraphDatabase类:get_entities函数:实体类型和属性都没提供，报错')

        if attributes == None:
            nodes = self.graph.nodes.match(entity_type).limit(limit).all()
        else:
            attributes1 = attributes[0]
            id = None
            if 'id' in attributes1:
                id = attributes1['id']
                attributes1.pop('id')
            if id == None:
                nodes = self.graph.nodes.match(entity_type,**attributes1).limit(limit).all()
            else:
                nodes = self.graph.nodes.match(entity_type, **attributes1).where('id(_)='+str(id)).limit(limit).all()

        if nodes == None:
            print('没有查询到实体')
            return None
        nodes2list = []
        for node in nodes:
            nodes2list.append(self._node_to_dict(node))
        return nodes2list

    def get_attr_value(self,aimed_attribute,entity_id,entity_name=None):
        # node_matcher = NodeMatcher(graph=self.graph)
        if entity_name == None:
            nodes = self.graph.nodes.match().where('id(_)=' + str(entity_id)).first()
        else:
            nodes = self.graph.nodes.match(name=entity_name).where('id(_)='+str(entity_id)).first()

        nodes_attr_dict = self._node_to_dict(nodes)
        if aimed_attribute in nodes_attr_dict.keys():
            return nodes_attr_dict[aimed_attribute]
        return False
        pass

    def get_car_attr_value(self,aimed_attribute,
                           car_series_name = None,car_series_id = None,
                           ):
        if car_series_name == None:
            if car_series_id == None:
                raise Exception('GraphDatabase类:get_car_attr_value函数:车型id和名字都没提供，报错')
            else:
                # car_node = self.graph.nodes.match().where('id(_)='+str(car_series_id)).first()
                car_id = car_series_id
        else:
            if car_series_id == None:
                car_node = self.graph.nodes.match(name=car_series_name).first()
            else:
                car_node = self.graph.nodes.match(name=car_series_name).where('id(_)=' + str(car_series_id)).first()

        if car_node == None:
            return False #id错误，车名错误，或者id和车名不匹配时，找不到对应车系，
        car_id = car_node.identity
        relations = self.graph.match().where('ID(a) ='+str(car_id)).all()
        for relation in relations:
            end_node_dict = self._node_to_dict(relation.end_node)
            # print(end_node_dict)
            if aimed_attribute in end_node_dict.keys():
                return end_node_dict[aimed_attribute]

        return None
        pass

    def query_relation2entity(self, entity, relation):
        entity_node = self.graph.nodes.match(name=entity).first()
        nodes = self.graph.match(nodes={entity_node}, r_type=relation).all()
        result  = []
        for node in nodes:
            result.append(node.start_node.get('name'))

        return result

    def query_attribute2entity(self, attributes):
        use_attributes = {}
        for k, v in attributes.items():
            if v is not None:
                use_attributes[slot_neo4j_dict[k]] = v

        base_parameters = self.graph.nodes.match('车型基本参数', **use_attributes).all()
        entities = []
        import sys
        for base_parameter in base_parameters:
            nodes = self.graph.match(nodes={base_parameter}, r_type='基本参数').all()
            for node in nodes:
                entities.append(node)

        result = []
        for entity in entities:
            result.append(entity.start_node.get('name'))

        return list(set(result))


    def query_entity2attribute(self, entity, relationship=None, c_attribute=None):
        e_attribute = c_attribute
        if relationship is not None and c_attribute is None:
            node = self.graph.nodes.match('车型', name=entity).first()
            r = self.graph.match(nodes={node}, r_type=relationship).first()
            return dict(r.end_node)

        if relationship is None and c_attribute is not None:
            # c_attribute 2 e_attribute
            for e, c in slot_neo4j_dict.items():
                if c_attribute == c:
                    e_attribute = e
                    break

            # 将e_attribute映射为属性顶点
            for k, v in schema.items():
                for vv in v['attribute']:
                    if vv == e_attribute:
                        relationship = k


            # 英文转中文
            relationship = slot_neo4j_dict[relationship]
            node = self.graph.nodes.match('车型', name=entity).first()
            r = self.graph.match(nodes={node}, r_type=relationship).first()

            result = {}
            result[c_attribute] = dict(r.end_node)[c_attribute]
            return result



