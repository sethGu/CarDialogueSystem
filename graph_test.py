from graph_database import GraphDatabase
from py2neo import Graph,Node,data,Path, Relationship
graphDatabase = GraphDatabase()

attributes = [{'name':'2021款 精英型'}]
result = graphDatabase.get_entities(entity_type='车型',attributes=attributes)
print(result) #[{'id': 1, 'label': '车型', 'name': '2021款 精英型'}]
attributes = [{'id':1}]
result = graphDatabase.get_entities(entity_type='车型',attributes=attributes)
print(result) #[{'id': 1, 'label': '车型', 'name': '2021款 精英型'}]

result = graphDatabase.get_entities(entity_type='车系')
print(result) #[{'id': 1, 'label': '车型', 'name': '2021款 精英型'}]
for i, e in enumerate(result):
    print(f"{i + 1}: {e['name']}")
#
# result = graphDatabase.get_entities(entity_type='车型')
# print(result) #[{'id': 955, 'label': '车型', 'name': '2021款 精英型'}, {'id': 963, 'label': '车型', 'name': '2019款 2.4L汽油手动四驱高底盘先锋版长厢'}......]
#
# result = graphDatabase.get_entities(entity_type='车身',limit=2)
# print(result) #[{'id': 957, 'label': '车身', '车身结构': '皮卡', '车门数(个)': 4.0, '高度(mm)': 1809.0, '座位数(个)': '5', '后排车门开启方式': '平开门', 'name': '车身节点', '轴距(mm)': 3155.0, '货箱尺寸(mm)': '1485x1510x530'}, {'id': 965, 'label': '车身', '车身结构': '皮卡', '车门数(个)': 4.0, '高度(mm)': 1809.0, '座位数(个)': '5', '后排车门开启方式': '平开门', 'name': '车身节点', '轴距(mm)': 3470.0, '货箱尺寸(mm)': '1800x1510x530'}]
#
# result = graphDatabase.get_entities(entity_type='发动机',limit=2)
# print(result) #[{'id': 961, 'label': '发动机', '最大扭矩(N·m)': '310', '最大功率(kW)': 130.0, 'name': '发动机节点'}, {'id': 968, 'label': '发动机', '每缸气门数(个)': '4', '供油方式': '多点电喷', '排量(L)': '2.4', '进气形式': '自然吸气', '最大扭矩(N·m)': '200', '环保标准': '国V', '最大功率(kW)': 105.0, '最大功率转速(rpm)': '5250', '最大马力(Ps)': 143.0, '最大扭矩转速(rpm)': '2500-3000', '缸体材料': '未知', '气缸数(个)': '4', '气缸排列形式': 'L', '缸盖材料': '铝合金', 'name': '发动机节点', '燃料形式': '汽油', '燃油标号': '92号', '配气机构': '未知'}]
#
# attributes = [{'供油方式':'直喷','排量(L)':'2.0'}]
# result = graphDatabase.get_entities(entity_type='发动机',attributes=attributes,limit=2)
# print(result) #[{'id': 1029, 'label': '发动机', '进气形式': '涡轮增压', '每缸气门数(个)': '4', '供油方式': '直喷', '排量(L)': '2.0', '最大扭矩(N·m)': '375', '环保标准': '国VI', '最大功率(kW)': 120.0, '最大马力(Ps)': 163.0, '缸体材料': '未知', '气缸排列形式': 'L', '气缸数(个)': '4', '缸盖材料': '未知', 'name': '发动机节点', '燃料形式': '柴油', '燃油标号': '0号', '配气机构': '未知'}, {'id': 1039, 'label': '发动机', '进气形式': '涡轮增压', '每缸气门数(个)': '4', '供油方式': '直喷', '排量(L)': '2.0', '最大扭矩(N·m)': '350', '环保标准': '国VI', '最大功率(kW)': 160.0, '最大马力(Ps)': 218.0, '缸体材料': '未知', '气缸排列形式': 'L', '气缸数(个)': '4', '缸盖材料': '铝合金', 'name': '发动机节点', '燃料形式': '汽油', '燃油标号': '92号', '配气机构': '未知'}]
#
# entity_id = 1029
# aimed_attribute = '供油方式'
# result = graphDatabase.get_attr_value(entity_id=entity_id,aimed_attribute=aimed_attribute)
# print(result)
#
# aimed_attribute = 'name'
# result = graphDatabase.get_attr_value(entity_id=entity_id,aimed_attribute=aimed_attribute)
# print(result)
#
# aimed_attribute = 'id'
# result = graphDatabase.get_attr_value(entity_id=entity_id,aimed_attribute=aimed_attribute)
# print(result)
#
# aimed_attribute = '高度(mm)'
# result = graphDatabase.get_attr_value(entity_id=entity_id,aimed_attribute=aimed_attribute)
# print(result)
#
# result = graphDatabase.get_car_attr_value(aimed_attribute=aimed_attribute,car_series_name='2021款 精英型')
# print(result)
#
# aimed_attribute = '助力类型'
# result = graphDatabase.get_car_attr_value(aimed_attribute=aimed_attribute,car_series_name='2017款 2.4L汽油手动两驱低底盘舒享型标厢')
# print(result)



# graph = Graph('http://localhost:7474', username='neo4j', password='12345')
# result = graph.nodes.match(labels='车型').limit(10).all()
# print(result)