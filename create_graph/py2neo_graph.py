# coding:utf-8
from py2neo import Graph,Node,data,Path, Relationship
from py2neo.matching import \
    NodeMatcher, RelationshipMatcher, \
    EQ, NE, LT, LE, GT, GE, \
    STARTS_WITH, ENDS_WITH, CONTAINS, LIKE, \
    IN, AND, OR, XOR

import pandas as pd


graph = Graph('http://localhost:7474', username='neo4j', password='123456')
graphMather = RelationshipMatcher(graph=graph)

data = pd.read_excel('./data/test3.xlsx',index_col=0)

for index in data.index:
    content = data.loc[index].dropna()

    #### 创建车系节点 ##########################################################
    car_series_node = graph.nodes.match('车系',name=content['车系1']).first()
    if car_series_node == None:
        car_series_node = Node('车系', name=content['车系1'])
        graph.create(car_series_node)
    ##############################################################

    #### 创建车型节点 ###########################################
    car_model_node = Node('车型', name=content['车型'])
    graph.create(car_model_node)
    ###############################################

    #### 创建车系车型边######################################
    r1 = Relationship(car_model_node, '属于', car_series_node)
    graph.create(r1)
    ########################################################

    #### 创建基本参数节点 ###########################################
    basic_parameters_dict = dict()
    basic_parameters_name = '基本参数节点'
    basic_parameters = ['能源类型','上市时间','最大功率(kW)',
                        '车身结构','最高车速(km/h)','工信部综合油耗(L/100km)',
                        '整车质保']
    for li in basic_parameters:
        try:
            if li == '上市时间':
                year = int(content[li])
                # print(year)
                # print(type(year))
                # year = int(year)
                basic_parameters_dict[li] = year
                continue
            basic_parameters_dict[li] = content[li]
        except:
            basic_parameters_dict[li] = '-'
            print('基本参数节点 报错\t',content['车型'],li)
    #######################
    cmb_node = graph.nodes.match('车型基本参数',**basic_parameters_dict).first()
    if cmb_node == None:
        # cmb_node = Node('车型基本参数', name=content['车型'], **basic_parameters_dict)
        cmb_node = Node('车型基本参数', name=basic_parameters_name, **basic_parameters_dict)
        graph.create(cmb_node)
    ########################################################

    #### 创建车型与基本参数的边 ###########################################
    r2 = Relationship(car_model_node, '基本参数', cmb_node)
    graph.create(r2)
    ########################################################

    #### 创建车身节点 ###########################################
    car_body_dict = dict()
    car_body_name = '车身节点'
    car_body = ['高度(mm)', '轴距(mm)', '前轮距(mm)', '后轮距(mm)',
                '最小离地间隙(mm)', '整备质量(kg)', '车身结构',
                '车门数(个)', '座位数(个)', '后排车门开启方式',
                '油箱容积(L)', '货箱尺寸(mm)']
    for li in car_body:
        try:
            car_body_dict[li] = content[li]
        except:
            car_body_dict[li] = '-'
            print('车身节点 报错\t',content['车型'],li)
    ####
    car_body_node = graph.nodes.match('车型车身',**car_body_dict).first()
    if car_body_node == None:
        car_body_node = Node('车型车身',name=car_body_name,**car_body_dict)
        graph.create(car_body_node)
    ########################################################

    #### 创建车型与车系的边 ###########################################
    r3 = Relationship(car_model_node,'车身',car_body_node)
    graph.create(r3)
    ########################################################

    #### 变速箱 ###########################################
    #通过对 变速箱、变速箱类型、简称 这3列的分析，单独某列无法作为节点的名称
    # 因此统一设置成一个名字
    transmission_case_name = '变速箱节点'
    transmission_case = ['变速箱类型','变速箱档位','变速箱简称']
    transmission_case_dict = dict()
    for li in transmission_case:
        try:
            transmission_case_dict[li] = content[li]
        except:
            transmission_case_dict[li] = '-'
            print('变速箱节点 报错\t',content['车型'],li)
    transmission_case_node = graph.nodes.match('变速箱',**transmission_case_dict).first()
    if transmission_case_node == None:
        transmission_case_node = Node('变速箱',name=transmission_case_name,**transmission_case_dict)
        graph.create(transmission_case_node)
    else:
        print('***'*20)
    ########################################################

    #### 创建车型与变速箱的边 ###########################################
    r4 = Relationship(car_model_node,'变速箱',transmission_case_node)
    graph.create(r4)
    ########################################################

    #### 底盘转向 ###########################################
    chassis_steering=['前悬架类型', '后悬架类型', '助力类型', '车体结构']
    chassis_steering_name = '底盘转向节点'
    chassis_steering_dict = dict()
    for li in chassis_steering:
        try:
            chassis_steering_dict[li] = content[li]
        except:
            chassis_steering_dict[li] = '-'
            print('底盘转向节点 报错\t',content['车型'],li)
    chassis_steering_node = graph.nodes.match('底盘转向',**chassis_steering_dict).first()
    if chassis_steering_node == None:
        chassis_steering_node = Node('底盘转向',name=chassis_steering_name,**chassis_steering_dict)
        graph.create(chassis_steering_node)
    else:
        print('***'*20)
    ########################################################

    #### 创建车型与底盘转向的边 ###########################################
    r5 = Relationship(car_model_node,'底盘转向',chassis_steering_node)
    graph.create(r5)
    ########################################################

    #### 车轮制动 ###########################################
    wheel_brake = ['前轮胎规格', '后轮胎规格']
    wheel_brake_name = '车轮制动节点'
    wheel_brake_dict = dict()
    for li in wheel_brake:
        try:
            wheel_brake_dict[li] = content[li]
        except:
            wheel_brake_dict[li] = '-'
            print('车轮制动节点 报错\t',content['车型'],li)
    wheel_brake_node = graph.nodes.match('车轮制动',**wheel_brake_dict).first()
    if wheel_brake_node == None:
        wheel_brake_node = Node('车轮制动',name=wheel_brake_name,**wheel_brake_dict)
        graph.create(wheel_brake_node)
    else:
        print('***'*20)
    ########################################################

    #### 创建车型与车轮制动的边 ###########################################
    r6 = Relationship(car_model_node,'车轮制动',wheel_brake_node)
    graph.create(r6)
    ########################################################

    #### 发动机 ###########################################
    engine = [ '排量(L)', '进气形式', '气缸排列形式', '气缸数(个)', '每缸气门数(个)',
               '压缩比', '配气机构', '缸径(mm)','行程(mm)',
               '最大马力(Ps)', '最大功率(kW)', '最大功率转速(rpm)',
               '最大扭矩(N·m)', '最大扭矩转速(rpm)',
               '燃料形式', '燃油标号', '供油方式',
               '缸盖材料', '缸体材料', '环保标准']
    engine_name = '发动机节点'
    engine_dict = dict()
    for li in engine:
        try:
            engine_dict[li] = content[li]
        except:
            engine_dict[li] = '-'
            print('发动机节点 报错\t',content['车型'],li)
    engine_node = graph.nodes.match('发动机',**engine_dict).first()
    if engine_node == None:
        engine_node = Node('发动机',name=engine_name,**engine_dict)
        graph.create(engine_node)
    else:
        print('***'*20)
    ########################################################

    #### 创建车型与发动机的边 ###########################################
    r7 = Relationship(car_model_node,'发动机',engine_node)
    graph.create(r7)
    ########################################################




