schema = {
    "car_series":{
        "attribute":['name'],
        "key":'name',
        "representation":['name'],
    },
    "car_model":{
        "attribute":["name"],
        "key":'name',
        "representation":['name'],
    },
    "basic_parameters":{
        "attribute":[
            'energy_type', #能源类型
            'time2market', #上市时间
            "maximum_power", #最大功率(kW)
            "body_structure", #车身结构
            "maximum_speed", #最高车速(km/h)
            "fuel_consumption", #工信部综合油耗(L/100km)
            "vehicle_warranty" #整车质保
        ],
        "key":"id",
        "representation":['energy_type','time2market',
                          'maximum_power','body_structure','maximum_speed',
                          'fuel_consumption','vehicle_warranty']
    },
    "car_body":{
        "attribute":[
            "height", #高度(mm)
            "wheelbase", #轴距(mm)
            "front_tract", #前轮距(mm)
            "rear_track", #后轮距(mm)
            "minimum_ground_clearance", #最小离地间隙(mm)
            "curb_weight", #整备质量(kg)
            "body_structure", #车身结构
            "door_num", #车门数(个)
            "seat_num", #座位数(个)
            "rear_door_open_mode", #后排车门开启方式
            "tank_volume", #油箱容积(L)
            "tank_volume" #货箱尺寸(mm)
        ],
        "key":'id',
        'representation':['height','wheelbase','front_tract','rear_track',
                          'minimum_ground_clearance','curb_weight','body_structure',
                          'door_num','seat_num','rear_door_open_mode',
                          'tank_volume','tank_volume'],
    },
    'transmission_case':{
        "attribute":[
            "transmission_type", #变速箱类型
            "transmission_gear", #变速箱档位
            "transmission_abbreviated" #变速箱简称
        ],
        "key":"id",
        "representation":[
            'transmission_type','transmission_gear','transmission_abbreviated'
        ],
    },
    'chassis_steering':{
        "attribute":[
            "f_suspension_type", #前悬架类型
            "r_suspension_type", #后悬架类型
            "type_of_assistance", #助力类型
            "car_body_structure" #车体结构
        ],
        "key":"id",
        "representation":[
            'f_suspension_type','r_suspension_type',
            'type_of_assistance','car_body_structure'
        ],
    },
    "wheel_brake":{
        "attribute":[
            "f_tire_size", #前轮胎规格
            "r_tire_size" #后轮胎规格
        ],
        "key":"id",
        "representation":[
            'f_tire_size','r_tire_size'
        ]
    },
    "engine":{
        "attribute":[
            "displacement", #排量(L)
            "air_intake_form", #进气形式
            "cylinder_arrangement", #气缸排列形式
            "cylinder_num", #气缸数(个)
            "valve_train", #每缸气门数(个)
            "compresstion_ratio", #压缩比
            "valve_system", #配气机构
            "cylinder_bore", #缸径(mm)
            "distance_of_run", #行程(mm)
            "maximum_horsepower", #最大马力(Ps)
            "maximum_power", #最大功率(kW)
            "maximum_power_speed", #最大功率转速(rpm)
            "maximum_torque", #最大扭矩(N·m)
            "maximum_torch_speed", #最大扭矩转速(rpm)
            "fuel_form", #燃料形式
            "fuel_grade", #燃油标号
            "oil_supply_mode", #供油方式
            "cylinder_head_material", #缸盖材料
            "cylinder_material", #缸体材料
            "environmental_standards", #环保标准
        ],
        "key":"id",
        "representation":['displacement','cylinder_num','valve_train',
                          'maximum_horsepower','maximum_power',
                          'fuel_form','fuel_grade','oil_supply_mode',
                          'environmental_standards'],
    }
}

slot_neo4j_dict = {
    'id':'id',
    'car_series':'车系',
    'car_model':'车型',
    'basic_parameters':'基本参数',
    'energy_type':'能源类型',
    'time2market':'上市时间',
    'maximum_power':'最大功率(kW)',
    'body_structure':'车身结构',
    'maximum_speed':'最高车速(km/h)',
    'fuel_consumption':'工信部综合油耗(L/100km)',
    'vehicle_warranty':'整车质保',
    'car_body':'车身',
    'height':'高度(mm)',
    'wheelbase':'轴距(mm)',
    'front_tract':'前轮距(mm)',
    'rear_track':'后轮距(mm)',
    'minimum_ground_clearance':'最小离地间隙(mm)',
    'curb_weight':'整备质量(kg)',
    'door_num':'车门数(个)',
    'seat_num':'座位数(个)',
    'rear_door_open_mode':'后排车门开启方式',
    'tank_capacity':'油箱容积',
    'tank_volume':'货箱尺寸(mm)',
    'transmission_case':'变速箱',
    'transmission_type':'变速箱类型',
    'transmission_gear':'变速箱档位',
    'transmission_abbreviated':'变速箱简称',
    'chassis_steering':'底盘转向',
    'f_suspension_type':'前悬架类型',
    'r_suspension_type':'后悬架类型',
    'type_of_assistance':'助力类型',
    'car_body_structure':'车体结构',
    'wheel_brake':'车轮制动',
    'f_tire_size':'前轮胎规格',
    'r_tire_size':'后轮胎规格',
    'engine':'发动机',
    'displacement':'排量(L)',
    'air_intake_form': '进气形式',
    'cylinder_arrangement': '气缸排列形式',
    'cylinder_num': '气缸数(个)',
    'valve_train': '每缸气门数(个)',
    'compresstion_ratio': '压缩比',
    'valve_system': '配气机构',
    'cylinder_bore': '缸径(mm)',
    'distance_of_run': '行程(mm)',
    'maximum_horsepower': '最大马力(Ps)',
    'maximum_power_speed': '最大功率转速(rpm)',
    'maximum_torque': '最大扭矩(N·m)',
    'maximum_torch_speed': '最大扭矩转速(rpm)',
    "fuel_form": "燃料形式",
    "fuel_grade": "燃油标号",
    "oil_supply_mode": "供油方式",
    "cylinder_head_material": "缸盖材料",
    "cylinder_material": "缸体材料",
    "environmental_standards": "环保标准",
}