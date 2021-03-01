import pandas as pd
import sys
# 节点列
nodes = ['车系1', '车型', '基本参数', '车身', '变速箱', '底盘转向', '车轮制动', '发动机']
# 基本参数
base_parameters = ['厂商', '级别', '能源类型', '环保标准', '上市时间', '最大功率(kW)',
                   '发动机', '变速箱',  '车身结构', '最高车速(km/h)', '工信部综合油耗(L/100km)', '整车质保']
# 车身
car_body = ['高度(mm)', '轴距(mm)', '前轮距(mm)', '后轮距(mm)', '最小离地间隙(mm)', '整备质量(kg)', '车身结构',
            '车门数(个)', '座位数(个)', '后排车门开启方式', '油箱容积(L)', '货箱尺寸(mm)']
# 变速箱
bian_su_xiang = ['简称', '变速箱类型']
# 底盘转向
di_pan_zhuan_xiang = ['前悬架类型', '后悬架类型', '助力类型', '车体结构']
# 车轮制动
che_lun_zhi_dong = ['前轮胎规格', '后轮胎规格']
# 发动机
fa_dong_ji = [ '排量(L)', '进气形式', '气缸排列形式', '气缸数(个)', '每缸气门数(个)', '压缩比', '配气机构', '缸径(mm)',
               '行程(mm)', '最大马力(Ps)', '最大功率(kW)', '最大功率转速(rpm)', '最大扭矩(N·m)', '最大扭矩转速(rpm)',
               '燃料形式', '燃油标号', '供油方式', '缸盖材料', '缸体材料', '环保标准']
# 文件名mapping
filenames_mapping = {'车系1':'che_xi', '车型':'che_xing', '基本参数':'base_parameters', '车身':'car_body', '变速箱':'bian_su_xiang',
                     '底盘转向':'di_pan_zhuan_xiang', '车轮制动':'che_lun_zhi_dong', '发动机':'fa_dong_ji'}


if __name__ == '__main__':
    data = pd.read_excel('./data/test3.xlsx', index_col='Unnamed: 0')

    # 车系和车型节点
    current_index = 0
    for node in nodes[:2]:
        result = pd.DataFrame(columns=['id:ID', 'name'])
        temp = list(set(data[node]))
        result['name'] = temp
        result['id:ID'] = [i for i in range(current_index, current_index + len(temp))]
        current_index += len(temp)
        # to csv
        result.to_csv('./data/entities/' + filenames_mapping[node] + '.csv', encoding='utf-8', index=False)

    che_xi = pd.read_csv('./data/entities/che_xi.csv')
    che_xing = pd.read_csv('./data/entities/che_xing.csv')
    # 车系和id 映射字典
    che_xi_mapping = {}
    for i, j in zip(che_xi['id:ID'], che_xi['name']):
        che_xi_mapping[j] = i
    # 车型和id 映射字典
    che_xing_mapping = {}
    for i, j in zip(che_xing['id:ID'], che_xing['name']):
        che_xing_mapping[j] = i

    # 车系和车型关系  一个车型对应一个车系id
    relationship = pd.DataFrame(columns=[':START_ID', ':END_ID', 'LINK'])
    relationship[':END_ID'] = data['车型'].map(che_xing_mapping)
    relationship[':START_ID'] = data['车系1'].map(che_xi_mapping)
    relationship['LINK'] = ['包含' for i in range(len(che_xing))]
    # 保存关系
    relationship.to_csv('./data/relationships/chexi2chexing.csv', encoding='utf-8', index=False)

    # other nodes
    schemas = zip(nodes[2:], [base_parameters, car_body, bian_su_xiang, di_pan_zhuan_xiang, che_lun_zhi_dong, fa_dong_ji])

    for k, v in schemas:
        # 配置 节点
        result = pd.DataFrame(columns=['车型'] + v)
        for cx in che_xing['name']:
            temp = data.loc[data['车型'] == cx][['车型'] + v]
            result = result.append(temp, ignore_index=True)

        result.insert(1, 'name', [k] * len(result))
        result.insert(1, 'id:ID', [i for i in range(current_index, current_index + len(result))])
        current_index = current_index + len(result)

        # 计算 车型 和 配置 之间的关系
        result['车型'] = result['车型'].map(che_xing_mapping)

        relationship = pd.DataFrame(columns=[':START_ID', ':END_ID', 'LINK'])
        relationship[':START_ID'] = result['车型']
        relationship[':END_ID'] = result['id:ID']
        relationship['LINK'] = ['配置' for i in range(len(result))]

        # 生成结果 result要删掉 车型列
        result.drop(columns='车型', inplace=True)
        result.to_csv('./data/entities/' + filenames_mapping[k] + '.csv', encoding='utf-8', index=False)
        relationship.to_csv('./data/relationships/che_xing2' + filenames_mapping[k] + '.csv', encoding='utf-8', index=False)

    # 品牌 节点
    result = pd.DataFrame(columns=['id:ID', 'name'])
    result['id:ID'] = [current_index]
    result['name'] = ['上汽大通']
    result.to_csv('./data/entities/brand.csv', encoding='utf-8', index=False)
    # 大通 和 车系 的关系, 一个车系对应一个品牌
    relationship = pd.DataFrame(columns=[':START_ID', ':END_ID', 'LINK'])
    relationship[':START_ID'] = [current_index for i in range(len(che_xi))]
    relationship[':END_ID'] = che_xi['id:ID']
    relationship['LINK'] = ['包含' for i in range(len(che_xi))]
    relationship.to_csv('./data/relationships/datong2chexi.csv', encoding='utf-8', index=False)














