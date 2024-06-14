import requests
import json

def get_travel_time(origin, destination, key):
    url = "https://restapi.amap.com/v3/direction/transit/integrated"
    parameters = {
        'key': key,
        'origin': origin,
        'destination': destination,
        'city': '北京',
        'cityd': '北京'
    }
    response = requests.get(url, params=parameters)
    data = json.loads(response.text)

    # 解析返回的数据
    if data['status'] == '1':
        route = data['route']
        if 'transits' in route:
            transits = route['transits']
            if transits:
                # 获取第一条路线的行程时间
                duration = transits[0]['duration']
                return duration
    return None

# 使用你的高德地图API密钥替换这里的'your_key'
key = '072cc54a9afeb80393b90f7540e99e4e'

# 获取两个地铁站的经纬度
origin = '116.337581,39.993138'  # 西二旗地铁站
destination = '116.434446,39.90816'  # 海淀黄庄地铁站

# 获取行程时间
duration = get_travel_time(origin, destination, key)
if duration:
    print(f'行程时间：{duration}秒')
else:
    print('无法获取行程时间')