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
