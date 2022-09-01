"""
前言：
由于原始方案中的自动特征工程代码跑一次 特征全搜索(融合600多个特征) 就要近一个小时
而且用于传统模型上数据的特征并非全自动提取
结合已有的经验和前人的总结，从这600多个特征中筛选出50多个基础特征和xx个额外特征
其中非传统模型(深度神经网络)训练时使用基础特征，而传统模型则使用 基础特征+额外特征 的方式进行训练
最后进行复现的文件中，特征工程代码结合之前筛选出的特征，然后更换成了传统方法
不过避免写重复代码，加了一个可以解析 FeaturesBaseRaw|FeaturesExtra 的函数—— SetHelper.parse_features

测试指导：
将原数据集(ccf_offline_stage1_test_revised.csv, ccf_offline_stage1_train.csv, ccf_online_stage1_train.csv)
放入一个文件夹中，文件夹路径记作 set_dir，使用 data_process(set_dir, save_dir) 进行数据处理，生成的新的数据集会放到运行目录下的 save_dir 中

"""

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random

# 辅助生成基础特征
# 以 [空格]drop 结尾的是辅助特征，会被删除
# 大致格式 set_type(-limit):{pivots_columns:agg_function}( drop)
#    其中：() 中是可选成分，{} 中是可重复成分
#         set_type: groupby 操作 作用的数据集，可选值有 self(标签集)、off1(线下特征集1)、off2(线下特征集2)、on(线上特征集)
#         limit：数据集的一些限制，比如 off1-normal_consume 表示 线下特征集1 中取正常消费(没使用优惠券消费)的部分，可选值参考数据集可独热的列
#                可以在前面加 ! 表示相反的
#         pivots_columns: 由一些基础列组成，可以是单一列，也可以是一个列表，以`-`隔开，表示 groupby 作用的轴
#         agg_function: 表示 groupby 之后的聚合操作，可以是 count(数量)，或者 column-function，例如 Distance-mean，对距离求均值
#                       当有多组 pivots_columns:agg_function 时会将上一组的输出作为下一组的输入进行新的操作
FeaturesBaseRaw = [
    'self:Merchant_id-Coupon_id:count',  # o14  商户-优惠券 维度
    'self:Merchant_id:count',  # o13  商户 维度
    'self:Merchant_id-User_id:count:Merchant_id:count',  # o15  商户 维度
    'self:User_id-Merchant_id:count',  # o8  用户-商户 维度
    'self:User_id-Coupon_id:count:User_id:count',  # o12  用户 维度
    'self:User_id:Date_received-max drop',
    'self:User_id:Date_received-min drop',
    'self:User_id:count',  # o1 用户 维度
    'self-!no_distance:Coupon_id:Distance-mean',  # 优惠券 维度
    'self-!no_distance:User_id:Distance-mean',  # 用户 维度
    'self:Coupon_id-Date_received:count',  # c5  优惠券 维度
    'self:Coupon_id-Date_received:count:Coupon_id:count',  # 优惠券 维度
    'self-!no_distance:Coupon_id-Date_received:Distance-mean',  # drop ? 优惠券 维度
    'self:Coupon_id:count',  # 优惠券 维度
    'self:Merchant_id-Date_received:count',  # m5  商户 维度
    'self:weekday:count',  # 日期 维度  （KFC 都有疯狂星期四，为什么优惠券就不可以有呢？
    'self:day:count',  # 日期 维度
    'self:User_id-weekday:count',  # 用户-日期 维度
    'self:Merchant_id-weekday:count',  # 商户-日期 维度

    # 过去长时间内
    'off1-!no_distance:Merchant_id:Distance-mean',  # 商户 维度
    'off1-!no_distance:User_id:Distance-mean',  # 用户 维度
    'off1-normal_consume:User_id:Date-max drop',
    'off1-normal_consume:User_id:Date-min drop',
    'off1-normal_consume:User_id:count',  # u5-1  用户 维度
    'off1-!no_consume:Merchant_id:count',  # m0-1  商户 维度
    'off1:Discount_rate:count',  # c8-1  优惠券 维度
    'off1-coupon_consume:Discount_rate:count',  # c9-1  优惠券 维度
    'off1-coupon_consume:Merchant_id:Date-max drop',
    'off1-coupon_consume:Merchant_id:Date-min drop',
    'off1-coupon_consume:Merchant_id:count',  # m1  商户 维度
    'off1-!no_consume:User_id-Merchant_id:count',  # um6-1  用户-商户 特征

    # 过去短时间内
    'off2-!no_distance:Merchant_id:Distance-mean',
    'off2-normal_consume:User_id:Date-max drop',
    'off2-normal_consume:User_id:Date-min drop',
    'off2-normal_consume:User_id:count',  # u5-2
    'off2-!no_consume:Merchant_id:count',  # m0-2
    'off2:Discount_rate:count',  # c8-2
    'off2-coupon_consume:Discount_rate:count',  # c9-2
    'off2-!no_consume:User_id-Merchant_id:count',  # um6-2

    # 线上行为
    'on:User_id:count',  # on_u1
    'on-is_click:User_id:count',  # on_u2
    'on-normal_consume:User_id:count drop',
    'on-fixed_consume:User_id:count drop',
    'on-coupon_consume:User_id:count drop',
    'on-no_consume:User_id:count drop',
]

FeaturesExtraRaw = [
    'self:User_id-Coupon_id:count',  # o2
    'self:Merchant_id-Coupon_id:count:Merchant_id:count',

    'off1-!no_distance:Merchant_id:count drop',
    'off1-!no_distance:Merchant_id:Distance-sum drop',
    'off1-coupon_consume:Merchant_id:discount_rate-mean',  # m8
    'off1-!normal_consume:Merchant_id:count',  # m3
    'off1-normal_consume:Merchant_id:count',  # m2-1
    'off2-normal_consume:Merchant_id:count',  # m2-2

    'on:User_id-Merchant_id:count:User_id:count',
    'on:User_id-Coupon_id:count:User_id:count',
]

# 基础的特征，所有基模型共享
# 最后生成的全部特征
FeaturesBase = [
    'Distance',
    'no_distance',
    'is_full_discount',
    'discount_x',
    'discount_y',
    'discount_rate',
    'discount_type',
    'day',
    'weekday',
    'f0',  # o7
    'f1',  # u6-1
    'f2',  # u6-2
    'f3',  # o4
    'f4',  # o3
    'f5',  # o5
    'f6',  # o6
    'f7',  # o17
    'f8',  # o18
    'f9',  # c11-1
    'f10',  # m20
    'f11',  # on_u4
    'f12',  # on_u5
    'f13',  # on_u3
    'f14',  # on_u6
    'f15',  # on_u7
    *[i.replace(':', '-') for i in FeaturesBaseRaw if not i.endswith(' drop')]
]

# 额外的特征，决策树类模型传统模型使用
FeaturesExtra = [
    'f16',  # m21
    'f17',  # m4
    *[i.replace(':', '-') for i in FeaturesExtraRaw if not i.endswith(' drop')]
]

no_date = pd.to_datetime(0)


class SetHelper:
    """
    处理数据集的类
    封装了一些处理数据用到的方法
    """

    ONLINE = 1
    OFFLINE_TRAIN = 2
    OFFLINE_TEST = 3

    @staticmethod
    def load_set(path: str,
                 off_test='ccf_offline_stage1_test_revised.csv',
                 off_train='ccf_offline_stage1_train.csv',
                 on_train='ccf_online_stage1_train.csv'):
        path = os.fspath(path)
        return pd.read_csv(os.path.join(path, off_test), parse_dates=['Date_received']), \
               pd.read_csv(os.path.join(path, off_train), parse_dates=['Date', 'Date_received']), \
               pd.read_csv(os.path.join(path, on_train), parse_dates=['Date', 'Date_received'])

    @staticmethod
    def set_process_assertion(df_processed: pd.DataFrame, set_type):
        assert df_processed.notna().all().all()  # 检查是否是NA
        if set_type == SetHelper.ONLINE:
            assert (df_processed['coupon_consume'] + df_processed['fixed_consume']
                    + df_processed['normal_consume'] + df_processed['no_consume']
                    + df_processed['is_click'] == 1).all()  # 检查线上数据集 one hot
        elif set_type == SetHelper.OFFLINE_TRAIN:
            assert (df_processed['normal_consume'] + df_processed['coupon_consume']
                    + df_processed['no_consume'] == 1).all()  # 检查线下数据集 one hot

    @staticmethod
    def pre_process_offline(df: pd.DataFrame):
        """线下训练数据集，有 Date"""
        if 'Date' in df.columns:
            df['normal_consume'] = 0  # 加上是否是正常消费
            df.loc[df['Coupon_id'].isna() & df['Date'].notna(), 'normal_consume'] = 1
            df['coupon_consume'] = 0  # 是否是使用优惠券消费 (没有15天限制)
            df.loc[df['Coupon_id'].notna() & df['Date'].notna(), 'coupon_consume'] = 1
            df['no_consume'] = 0  # 领了优惠券但没有消费
            df.loc[df['Coupon_id'].notna() & df['Date'].isna(), 'no_consume'] = 1
            df['Coupon_id'] = df['Coupon_id'].fillna(0).astype(
                int)  # Coupon_id 由 nullable 转换成 notnull 会把整数类型转成 float，这里转回去
            df['Discount_rate'].fillna('1.0', inplace=True)  # 没有优惠券消费相当于10折，这里得填 str 下面类型才不会出问题
            # df['Distance'].fillna(-1, inplace=True)  Distance 下面就可以处理
            df['Date_received'].fillna(no_date, inplace=True)
            df['Date'].fillna(no_date, inplace=True)

        '''
        线下测试数据集处理，没有 Date
        '''
        df['Distance'] = df['Distance'].fillna(-1).astype(int)
        df['no_distance'] = (df['Distance'] == -1).astype(int)
        df = SetHelper.__process_discount(df)
        SetHelper.set_process_assertion(df, SetHelper.OFFLINE_TRAIN if 'Date' in df.columns else SetHelper.OFFLINE_TEST)
        return df

    @staticmethod
    def pre_process_online(df: pd.DataFrame):
        df['coupon_consume'] = 0  # 是否使用了优惠券消费
        df.loc[df['Date'].notna() & df['Coupon_id'].notna() & (df['Coupon_id'] != 'fixed'), 'coupon_consume'] = 1
        df['fixed_consume'] = 0  # 是否是限时降价消费
        df.loc[df['Date'].notna() & (df['Coupon_id'] == 'fixed'), 'fixed_consume'] = 1
        # 移除 fixed 的 Date_received
        df.loc[df['Date'].notna() & (df['Coupon_id'] == 'fixed'), 'Date_received'] = no_date
        df['normal_consume'] = 0  # 是否是正常消费，没有使用优惠券的消费行为
        df.loc[(df['Action'] == 1) & df['Coupon_id'].isna(), 'normal_consume'] = 1
        df['no_consume'] = 0  # 是否是领取了优惠券但没有消费
        df['is_click'] = (df['Action'] == 0).astype(int)  # 是否是点击行为
        df.loc[(df['Action'] == 2), 'no_consume'] = 1
        df['Date'].fillna(no_date, inplace=True)
        df['Date_received'].fillna(no_date, inplace=True)
        df['Coupon_id'] = df['Coupon_id'].replace('fixed', 0)
        df['Coupon_id'] = df['Coupon_id'].fillna(0).astype(int)
        df['Discount_rate'] = df['Discount_rate'].replace('fixed', '-1.0')  # 标记为 -1.0
        df['Discount_rate'].fillna('1.0', inplace=True)
        df = SetHelper.__process_discount(df)
        SetHelper.set_process_assertion(df, SetHelper.ONLINE)
        return df

    @staticmethod
    def __process_discount(df):
        df['is_full_discount'] = df['Discount_rate'].str.contains(':').astype(int)
        df[['discount_x', 'discount_y']] = df[df['is_full_discount'] == 1]['Discount_rate'] \
            .str.split(':', expand=True).astype(float)
        # expand 设置成 true 才可以返回一个 dataframe，设置成 float 是因为合并时有NA
        df['discount_rate'] = (1 - (df['discount_y'] / df['discount_x'])) \
            .fillna(df['Discount_rate']).astype(float)
        df[['discount_x', 'discount_y']] = \
            df[['discount_x', 'discount_y']].fillna(-1).astype(int)
        _rate = sorted(set(df.discount_rate))
        # 加上一个折扣券类型
        df['discount_type'] = df['discount_rate'].apply(lambda x: _rate.index(x))
        return df

    @staticmethod
    def split_dataset(on: pd.DataFrame, off: pd.DataFrame, test: pd.DataFrame):
        L_1_range = pd.date_range(start='20160516', end='20160615')  # 标签集1: 5.16-6.15(Date_received)

        F_1_1_range = pd.date_range(start='20160201', end='20160430')  # 特征集1-1: 2.1-4.30(Date_received|Date)
        F_1_2_range = pd.date_range(start='20160501', end='20160515')  # 特征集1-2: 5.1-5.15(Date)

        L_2_range = pd.date_range(start='20160415', end='20160515')  # 标签集2: 4.15-5.15(Date_received)

        F_2_1_range = pd.date_range(start='20160101', end='20160330')  # 特征集2-1: 1.1-3.30(Date_received|Date)
        F_2_2_range = pd.date_range(start='20160331', end='20160414')  # 特征集2-1: 3.31-4.14(Date)

        L_T_range = pd.date_range(start='20160701', end='20160731')  # 测试集: 7.1-7.31(Date_received)

        F_T_1_range = pd.date_range(start='20160318', end='20160615')  # 测试特征集1: 3.18-6.15(Date_received|Date)
        F_T_2_range = pd.date_range(start='20160616', end='20160630')  # 测试特征集2: 6.16-6.30(Date)

        # 检查分布是否一致
        assert L_1_range.size == L_2_range.size == L_T_range.size
        assert F_1_1_range.size == F_2_1_range.size == F_T_1_range.size
        assert F_1_2_range.size == F_2_2_range.size == F_T_2_range.size

        label_1 = off[off['Date_received'].isin(L_1_range)]
        f_1_1_off = off[off['Date_received'].isin(F_1_1_range) | off['Date'].isin(F_1_1_range)]
        f_1_2_off = off[off['Date'].isin(F_1_2_range)]  # 最近15天线下消费活动
        f_1_on = on[on['Date'].isin(F_1_2_range) | on['Date_received'].isin(F_1_1_range) | on['Date'].isin(F_1_1_range)]

        label_2 = off[off['Date_received'].isin(L_2_range)]
        f_2_1_off = off[off['Date_received'].isin(F_2_1_range) | off['Date'].isin(F_2_1_range)]
        f_2_2_off = off[off['Date'].isin(F_2_2_range)]  # 最近15天线下消费活动
        f_2_on = on[on['Date'].isin(F_2_2_range) | on['Date_received'].isin(F_2_1_range) | on['Date'].isin(F_2_1_range)]

        f_test_1_off = off[off['Date_received'].isin(F_T_1_range) | off['Date'].isin(F_T_1_range)]
        f_test_2_off = off[off['Date'].isin(F_T_2_range)]  # 最近15天线下消费活动
        f_test_on = on[
            on['Date'].isin(F_T_2_range) | on['Date_received'].isin(F_T_1_range) | on['Date'].isin(F_T_1_range)]

        return (label_1, f_1_1_off, f_1_2_off, f_1_on), (label_2, f_2_1_off, f_2_2_off, f_2_on), \
               (test, f_test_1_off, f_test_2_off, f_test_on)

    @staticmethod
    def parse_set_type(set_type, label_set, off_feat_1, off_feat_2, on_feat):
        if set_type == 'self':
            return label_set.copy(deep=False)
        elif set_type == 'off1':
            return off_feat_1.copy(deep=False)
        elif set_type == 'off2':
            return off_feat_2.copy(deep=False)
        elif set_type == 'on':
            return on_feat.copy(deep=False)
        else:
            set_type, limit = set_type.split('-')
            opposite = 1
            if limit[:1] == '!':
                opposite = 0
                limit = limit[1:]
            s = SetHelper.parse_set_type(set_type, label_set, off_feat_1, off_feat_2, on_feat)
            return s[s[limit] == opposite].copy(deep=False)

    @staticmethod
    def parse_features(feats, label_set, off_feat_1, off_feat_2, on_feat):
        # 解析 feats, 生成特征，有利于去除重复代码
        drop_feats = [i.replace(' drop', '').replace(':', '-') for i in feats if i.endswith(' drop')]
        base_feats = [i.replace(' drop', '').split(':') for i in feats if i.__contains__(':')]
        for op in base_feats:
            X = SetHelper.parse_set_type(op[0], label_set, off_feat_1, off_feat_2, on_feat)
            last = X
            last_on = []
            last_i = (len(op) - 1) // 2 - 1
            for i in range((len(op) - 1) // 2):
                pivots = op[2 * i + 1].split('-')
                agg = op[2 * i + 2]
                grped = last.groupby(pivots)
                if agg == 'count':
                    last = grped.size().reset_index(name='-'.join(op) if i == last_i else '')
                else:
                    col, func = agg.split('-')
                    last = grped.agg(func)[col].reset_index(name='-'.join(op) if i == last_i else '')
                if i == last_i:
                    last_on = pivots
            label_set = pd.merge(label_set, last, how='left', on=last_on)
            print(f'parsed feature {"-".join(op)} for {op[0]}.')
        return label_set, drop_feats

    @staticmethod
    def __window(arr, n=-1, slices=-1):
        if n == -1 and slices != -1:
            n = (len(arr) // slices) + 1
        if len(arr) <= n:
            return [arr]
        ret = []
        piece = len(arr) // n
        remain = len(arr) - (piece * n)
        for i in range(piece):
            ret.append(arr[i * n:(i + 1) * n])
        if remain > 0:
            ret.append(arr[len(arr) - remain:])
        return ret

    @staticmethod
    def task(chunk: pd.DataFrame, label_set: pd.DataFrame):
        chunk = chunk.copy(deep=False)

        for i, x in chunk.iterrows():
            tmp = label_set[label_set['User_id'] == x['User_id']]
            tmp1 = tmp['Date_received'] < x['Date_received']
            tmp2 = tmp['Date_received'] > x['Date_received']
            f3 = sum(tmp1.astype(int))
            f4 = sum(tmp2.astype(int))
            tmp1 = tmp[tmp1]
            tmp2 = tmp[tmp2]
            f5 = sum((tmp1['Coupon_id'] == x['Coupon_id']).astype(int))
            f6 = sum((tmp2['Coupon_id'] == x['Coupon_id']).astype(int))
            f7 = (x['Date_received'] - tmp1['Date_received'].max()).days
            f8 = (tmp2['Date_received'].min() - x['Date_received']).days
            chunk.loc[i, 'f3'] = f3
            chunk.loc[i, 'f4'] = f4
            chunk.loc[i, 'f5'] = f5
            chunk.loc[i, 'f6'] = f6
            chunk.loc[i, 'f7'] = f7
            chunk.loc[i, 'f8'] = f8

        return chunk

    @staticmethod
    def data_structure_base(label_set: pd.DataFrame, off_feat_1: pd.DataFrame,
                            off_feat_2: pd.DataFrame, on_feat: pd.DataFrame):
        """
        基础特征工程
        """
        label_set = label_set.copy(deep=False)
        # 日期
        label_set['weekday'] = label_set['Date_received'].dt.weekday
        label_set['day'] = label_set['Date_received'].dt.day

        # 自动融合部分
        label_set, drop_feats = SetHelper.parse_features(FeaturesBaseRaw, label_set, off_feat_1, off_feat_2, on_feat)

        print('parsed feature f0-f2')
        # 手动融合特征部分
        label_set['f0'] = (label_set['self-User_id-Date_received-max'] -
                           label_set['self-User_id-Date_received-min']).dt.days / \
                          (label_set['self-User_id-count'])
        label_set['f1'] = (label_set['off1-normal_consume-User_id-Date-max'] -
                           label_set['off1-normal_consume-User_id-Date-min']).dt.days / \
                          (label_set['off1-normal_consume-User_id-count'])
        label_set['f2'] = (label_set['off2-normal_consume-User_id-Date-max'] -
                           label_set['off2-normal_consume-User_id-Date-min']).dt.days / \
                          (label_set['off2-normal_consume-User_id-count'])

        num_workers = os.cpu_count() // 2  # 适当调大这个值可以加快处理，不过会吃满占用的核
        label_set_chunks = SetHelper.__window(label_set, slices=num_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            print('parsed feature f3-f8')
            label_set = pd.concat(exe.map(SetHelper.task, label_set_chunks,
                                          [label_set] * num_workers), axis=0)

        label_set['f9'] = label_set['off1-coupon_consume-Discount_rate-count'] / label_set['off1-Discount_rate-count']
        label_set['f10'] = (label_set['off1-coupon_consume-Merchant_id-Date-max'] -
                            label_set['off1-coupon_consume-Merchant_id-Date-min']).dt.days / \
                           (label_set['off1-coupon_consume-Merchant_id-count'] - 1)
        label_set['f11'] = label_set['on-normal_consume-User_id-count'] \
                           + label_set['on-fixed_consume-User_id-count'] \
                           + label_set['on-coupon_consume-User_id-count']
        label_set['f12'] = label_set['f11'] / label_set['on-User_id-count']
        label_set['f13'] = label_set['on-is_click-User_id-count'] / label_set['on-User_id-count']
        label_set['f14'] = label_set['on-no_consume-User_id-count'] + label_set['on-coupon_consume-User_id-count']
        label_set['f15'] = label_set['f14'] / label_set['on-User_id-count']

        # 移除标记特征
        label_set.drop(drop_feats, axis=1, inplace=True)
        print()
        return label_set.fillna(0)

    @staticmethod
    def data_structure_extra(label_set: pd.DataFrame, off_feat_1: pd.DataFrame,
                             off_feat_2: pd.DataFrame, on_feat: pd.DataFrame):
        """
        额外特征工程
        """
        label_set = label_set.copy(deep=False)
        label_set, drop_feats = SetHelper.parse_features(FeaturesExtraRaw, label_set, off_feat_1, off_feat_2, on_feat)

        label_set['f16'] = (label_set['off1-!no_distance-Merchant_id-Distance-sum']) / \
                           (label_set['off1-!no_distance-Merchant_id-count'])

        label_set['f17'] = (label_set['off1-coupon_consume-Merchant_id-count']) / \
                           (label_set['off1-!normal_consume-Merchant_id-count'])

        label_set.drop(drop_feats, axis=1, inplace=True)
        print()
        return label_set.fillna(0)

    @staticmethod
    def attach_labels(_label_set: pd.DataFrame) -> pd.DataFrame:
        """
        给 label_set 打上标签
        :param _label_set:
        :return: label_set
        """
        _label_set['label'] = 0
        _label_set.loc[(_label_set['Date'] != no_date) &  # 要求有 Date
                       (_label_set['Date'] - _label_set['Date_received'] <= pd.to_timedelta(15, 'D')),  # 并且小于 15 天
                       'label'] = 1
        return _label_set

    # 经简单测试发现过采样的效果并不好
    @staticmethod
    def SMOTE(label_set: pd.DataFrame, feat_cols, xx=1.0, n_sampling=0, noise_weight=0.1):
        """
        SMOTE 过采样，会直接生成用于训练的数据集
        :param feat_cols: 特征列
        :param label_set: 标签集
        :param n_sampling:  int，过采样的数量
        :param xx: float，>1，过采样后的倍率，会覆盖 n_sampling
        :param noise_weight: float，0-1，噪声的权重
        """
        X_min = label_set[feat_cols]
        y_min = label_set['label']
        if xx is not None:
            assert xx > 1.0
            n_sampling = int(len(X_min) * (xx - 1.0))
        # 数据集内找五个最近的数据，组成一个数组
        n_nearests = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree') \
            .fit(X_min).kneighbors(X_min)[1]
        # 全部置0，生成过采样的数据集的数据集
        X_res = np.zeros((n_sampling, X_min.shape[1]))
        y_res = np.zeros(n_sampling)
        for i in range(n_sampling):
            # 随机选取五个临近点
            reference = random.randint(0, len(n_nearests) - 1)
            # 五个点
            all_point = n_nearests[reference]
            ser = y_min[y_min.index.isin(all_point)].sum(skipna=True)
            y_res[i] = 1 if ser > 2 else 0  # 如果大于一半都是正例那就是正例
            # 随机选一个邻居点，第一个点是中心点，要去掉，所以是 1:
            neighbour = random.choice(n_nearests[reference, 1:])
            noise = random.random() * noise_weight  # 随机的噪声
            # 中心点减去一个随机的邻居点，作为距离
            gap = (X_min.loc[reference, :] - X_min.loc[neighbour, :])
            # 根据中心点生成一个新的数据
            X_res[i] = np.array(X_min.loc[reference, :] + noise * gap)
        X_res = pd.DataFrame(X_res, columns=X_min.columns)
        y_res = pd.Series(y_res, name=y_min.name, dtype=int)
        X_con = pd.concat([X_min, X_res], axis=0, ignore_index=True).astype(X_min.dtypes)
        y_con = pd.concat([y_min, y_res], axis=0, ignore_index=True).astype(y_min.dtypes)
        return pd.concat([X_con, y_con], axis=1)


class Stacking:
    """
    模型融合部分
    """

    def __init__(self, save_dir):
        df1 = pd.read_csv(os.path.join(save_dir, 'train/train_set_1_base.csv'),
                          parse_dates=['Date_received', 'Date'])
        df2 = pd.read_csv(os.path.join(save_dir, 'train/train_set_2_base.csv'),
                          parse_dates=['Date_received', 'Date'])
        self.train_set_base = pd.concat([df1, df2], axis=0, ignore_index=True)
        df3 = pd.read_csv(os.path.join(save_dir, 'train/train_set_1_extra.csv'),
                          parse_dates=['Date_received', 'Date'])
        df4 = pd.read_csv(os.path.join(save_dir, 'train/train_set_2_extra.csv'),
                          parse_dates=['Date_received', 'Date'])
        self.train_set_extra = pd.concat([df3, df4], axis=0, ignore_index=True)
        self.test_set = pd.read_csv(os.path.join(save_dir, 'test/test_set.csv'),
                                    parse_dates=['Date_received'])
        self.feat_cols_base = FeaturesBase
        self.feat_cols_extra = FeaturesBase + FeaturesExtra

    def train(self):
        pass

    def valid(self):
        pass

    def pred(self, save_to):
        pass


def data_process(set_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(f'{save_dir}/train')
        os.mkdir(f'{save_dir}/test')

    test, off, on = SetHelper.load_set(set_dir)

    print('start preprocess dataset.')
    test = SetHelper.pre_process_offline(test)
    off = SetHelper.pre_process_offline(off)
    on = SetHelper.pre_process_online(on)
    print('process succeeded.')

    print('start splitting dataset.')
    t1, t2, t3 = SetHelper.split_dataset(on, off, test)
    print('split succeeded.')

    train_set1 = SetHelper.data_structure_base(*t1)
    train_set1 = SetHelper.attach_labels(train_set1)
    train_set1.to_csv(f'./{save_dir}/train/train_set_1_base.csv', index=False)
    train_set1 = SetHelper.data_structure_extra(train_set1.drop(columns=['label']), *(t1[1:]))
    train_set1 = SetHelper.attach_labels(train_set1)
    train_set1.to_csv(f'./{save_dir}/train/train_set_1_extra.csv', index=False)

    train_set2 = SetHelper.data_structure_base(*t2)
    train_set2 = SetHelper.attach_labels(train_set2)
    train_set2.to_csv(f'./{save_dir}/train/train_set_2_base.csv', index=False)
    train_set2 = SetHelper.data_structure_extra(train_set2.drop(columns=['label']), *(t2[1:]))
    train_set2 = SetHelper.attach_labels(train_set2)
    train_set2.to_csv(f'./{save_dir}/train/train_set_2_extra.csv', index=False)

    test_set = SetHelper.data_structure_base(*t3)
    test_set = SetHelper.data_structure_extra(test_set, *(t3[1:]))
    test_set.to_csv(f'./{save_dir}/test/test_set.csv', index=False)


if __name__ == '__main__':
    data_process('dataset_raw', 'dataset_processed')
