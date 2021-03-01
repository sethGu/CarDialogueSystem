from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.matching import RelationshipMatcher
import sys

class GraphOp():
    def __init__(self):
        # connect database
        try:
            self.graph = Graph("http://localhost:7474", username='neo4j', password='123456')
            print("database connect successful!")
        except:
            print("database connect fail!")
            sys.exit()

    def query_entity(self):
        nmatcher = NodeMatcher(self.graph)
        rmatcher = RelationshipMatcher(self.graph)

        # 大通有哪些车
        a_node = nmatcher.match('品牌', name='上汽大通').first()
        b_node = nmatcher.match('车系').first()
        relation = rmatcher.match({a_node, b_node}).first()
        zz = self.graph.match((a_node,), r_type=type(relation).__name__).all()
        result = []
        for z in zz:
            result.append(z.end_node.get('name'))
        print('大通具有：')
        print(result)
        # nodes = rmatcher.match({a_node, b_node}).all()
        # print(nodes[0])
        # print(nodes[0].start_node)
        # print(nodes[0].end_node)
        # print(nodes[0].nodes)
        # print(type(nodes[0]).__name__)

        #
        # A(实体)车的基本参数(实体)
        a_node = nmatcher.match('车型', name='2019款 2.0T柴油自动四驱精英型长厢高底盘').first()
        b_node = nmatcher.match('基本参数').first()
        relation = rmatcher.match({a_node, b_node}).first()
        print(relation)
        print(type(relation).__name__)
        print(dict(nmatcher.match(name='2019款 2.0T柴油自动四驱精英型长厢高底盘').first()))
        zz = self.graph.match(nodes=(relation.start_node,), r_type=type(relation).__name__).all()[0].end_node
        print(zz)
        # print('*******')
        # gg = self.graph.match(nodes=(relation.start_node, relation.end_node)).all()
        # print(dict(gg[0].end_node))
        #
        # print('$$$$$$$')
        # for z in zz:
        #     if z.end_node.get('name') == '基本参数':
        #         property = dict(z.end_node.items())
        # print('基本参数：')
        # print(property)

        # A(实体)车的发动机(属性)怎么样

    def query_test(self):
        nmatcher = NodeMatcher(self.graph)
        rmatcher = RelationshipMatcher(self.graph)
        # A车的基本参数
        a_node = nmatcher.match('车型', name='2019款 2.0T柴油自动四驱精英型长厢高底盘').first()

        # relation = rmatcher.match({a_node, b_node}).first()
        relation = self.graph.match(nodes=(a_node,), r_type='基本参数').all()
        print(dict(relation[0].end_node))
        # gg = self.graph.match(nodes=(relation.start_node, relation.end_node)).all()

        # 
        b_node = nmatcher.match('车型基本参数', name='基本参数节点').first()










if __name__ == '__main__':
    graph_op = GraphOp()
    graph_op.query_entity()
