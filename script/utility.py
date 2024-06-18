import pandas as pd
import numpy as np
from typing import List

# 处理数据输入的cell

def readDatas(rawDatas: List[pd.DataFrame], csvList: List[str]) -> None:
    """将列出的所有csv文件 内容 按列表顺序存入rawDatas中"""
    for index in range(len(csvList)):
        tmpDf = pd.read_csv(csvList[index])
        rawDatas.append(tmpDf)
    return None
    
def  readDateList(dateList: List[str], csvList: List[str]) -> None:
    """将列出的所有csv文件所属 日期 按顺序存入dateList中"""
    for index in range(len(csvList)):
        tmpDate = csvList[index][-8:-4]
        dateList.append(tmpDate)
    return None

def get_rem_sta_info(origin_df: pd.DataFrame, rem_sta_list: list[str]) -> pd.DataFrame:
    """ 填补单个csv文件中缺失站点的数据 """
    date_str = origin_df.loc[0, '日期'] # 日期与文件中其他所有数据保持一致
    line_str = '' # 线路名不纳入训练，初始化为空字符串
    outflow = type(origin_df.loc[0, '进站量'])(0)
    inflow = type(origin_df.loc[0, '进站量'])(0)
    start_time_enum = list(origin_df['开始时间'].unique())
    end_time_enum = list(origin_df['结束时间'].unique())
    time_point_num = len(start_time_enum)

    output_df = pd.DataFrame(columns=origin_df.columns)
    for time_i in range(time_point_num):
        for rem_sta_i in range(len(rem_sta_list)):
            new_row = pd.DataFrame([{'日期':date_str,
                                     '开始时间':start_time_enum[time_i],
                                     '结束时间':end_time_enum[time_i],
                                     '站点名':rem_sta_list[rem_sta_i],
                                     '线路名':line_str,
                                     '进站量':inflow,
                                     '出站量':outflow
                                     }])
            output_df = pd.concat([output_df, new_row], ignore_index=True)
    return output_df


def get_station_for_adj(stop_for_adj: pd.DataFrame, csv_list: list[str]) -> pd.DataFrame:
    """ 输入邻接矩阵中所有的站点名以及要读入的csv文件名列表， 创建station的dataframe """
    tmp_df = pd.read_csv(csv_list[0])    # 获取列表中首个文件的数据

    # 添加文件数据中缺失的站点数据，目前方案为全部赋值为0
    remain_staion_list = [] # 缺失的站点名列表
    for station_name in list(stop_for_adj['站点名']):
        if not station_name in list(tmp_df['站点名']):
            remain_staion_list.append(station_name)
    print('remain_staion_list is ', len(remain_staion_list))

    output_df_list = []
    for csv_file in csv_list:
        tmp_ori_df = pd.read_csv(csv_file) # 获取原始数据
        output_df_list.append(tmp_ori_df)
        tmp_rem_df = get_rem_sta_info(tmp_ori_df, remain_staion_list) # 填补缺失数据
        output_df_list.append(tmp_rem_df)
        print('tmp_rem_df.__len__ is ', len(tmp_rem_df))

    output_df = pd.concat(output_df_list, axis=0, ignore_index=True)
    print('output_df.__len__ is ', len(output_df))
    return output_df

class cstRawCsvData:
    def __init__(self, stationCsvLists: List[str] = [], ODCsvLists: List[str] = []) -> None:
        """
        初始化输入: csv文件列表，包括station的和OD的
        而后将这些文件的内容读入相应的数据成员变量列表中，并将文件名中的日期读入日期成员变量列表中
        """

        self.rawStationDatas: List[pd.DataFrame] = []
        self.rawODDatas: List[pd.DataFrame] = []

        self.stationDateList: List[str] = []
        self.ODDateList: List[str] = []

        if stationCsvLists != None and stationCsvLists !=[]:
            readDatas(self.rawStationDatas, stationCsvLists)
            readDateList(self.stationDateList, stationCsvLists)
        else:
            self.rawStationDatas = []

        if ODCsvLists != None and ODCsvLists !=[]:
            readDatas(self.rawODDatas, ODCsvLists)
            readDateList(self.ODDateList, ODCsvLists)
        else:
            self.rawODDatas = []

        return None

    def getAllStationData(self) -> pd.DataFrame:
        if not self.rawStationDatas:
            print("empty station data, please check it")
            return None
        
        outputDf = pd.DataFrame()
        for df in self.rawStationDatas:
            outputDf = pd.concat([outputDf, df])
        return outputDf
    
    def getAllODData(self) -> pd.DataFrame:
        if not self.rawODDatas:
            print("empty OD data, please check it")
            return None
        outputDf = pd.DataFrame()
        for df in self.rawODDatas:
            outputDf = pd.concat([outputDf, df])
        return outputDf
    
    def getSpecDateDf(self, date: str, sta_or_od = 1) -> pd.DataFrame:
        """获取指定日期的数据, 以dataframe类型给出"""
        i = 0
        if (sta_or_od == 1):
            # 默认获取某一天的station数据
            for i in range(len(self.stationDateList)):
                if date == self.stationDateList[i]:
                    break
                if i == (len(self.stationDateList) - 1):
                    print("no such date info here, please check input csv list")
                    return None
            return self.rawStationDatas[i]
        else:
            # 获取某一天的OD数据
            for i in range(len(self.ODDateList)):
                if date == self.ODDateList[i]:
                    break
                if i == (len(self.ODDateList) - 1):
                    print("no such date info here, please check input csv list")
                    return None
            return self.rawODDatas[i]
        
# 给出对dataframe的一些基本操作接口

def getDfBySpecPeriod(startTime: str, endTime: str, originalDf: pd.DataFrame) -> pd.DataFrame:
    """根据所给的时间段，获取dataframe中时间片位于时间段内的条目，并组成新的dataframe作为输出"""
    startT, endT = pd.to_datetime(startTime), pd.to_datetime(endTime)
    outputDf = pd.DataFrame(columns=originalDf.columns)

    for index in range(len(originalDf)):
        tmpStart= pd.to_datetime(originalDf.loc[:, '开始时间'].iloc[index])
        tmpEnd = pd.to_datetime(originalDf.loc[:, '结束时间'].iloc[index])
        if tmpStart <= startT and tmpEnd >= endT:
            outputDf.loc[len(outputDf)] = originalDf.iloc[index]
            outputDf.reset_index()

    return outputDf

def getDfByStationNameList(stationNameList: List[str], originalDf: pd.DataFrame) -> pd.DataFrame:
    """根据所给的站名列表，获取originDf中相应站名的条目，组成新的dataframe输出"""
    outputDf = pd.DataFrame(columns=originalDf.columns)

    for index in range(len(originalDf)):
        tmpName = str(originalDf.loc[:, '站点名'].iloc[index])
        if tmpName in stationNameList:
            outputDf.loc[len(outputDf)] = originalDf.iloc[index]
            outputDf.reset_index()

    return outputDf
