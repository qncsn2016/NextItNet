import pandas as pd
import numpy as np
import time


def yoosortlen(from_path):
    file = open(from_path, "r")
    lines = file.readlines()

    pad_idx = 0
    items2idx = {}  # the ret
    items2idx['<pad>'] = pad_idx
    idx_cnt = 0

    lendict = dict()
    for i in range(3, 10):
        lendict['%d' % i] = []
    for line in lines:
        linelist = line.splitlines()[0].split(',')
        ids = []
        for i in range(len(linelist)):
            item=int(linelist[i])
            if item not in items2idx:
                if idx_cnt == pad_idx:
                    idx_cnt += 1
                items2idx[item] = idx_cnt
                idx_cnt += 1
            ids.append(items2idx[item])
        lendict['%d' % len(ids)].append(ids)

    train_dict = dict()
    test_dict = dict()
    for i in range(3, 10):
        dev_sample_index = -1 * int(0.2 * float(len(lendict['%d' % i])))
        train_dict['%d' % i] = lendict['%d' % i][:dev_sample_index]
        test_dict['%d' % i] = lendict['%d' % i][dev_sample_index:]

    # shuffle?

    return train_dict, test_dict, items2idx

# ---wjy
def tosessions_yoo(from_path, to_path):
    data = pd.read_csv(from_path, sep=',', dtype={'ItemId': np.int64})
    print("read finish from " + from_path)
    data.sort_values(['SessionId', 'Time'], inplace=True)
    print("sort finish")
    sessionid = list(data['SessionId'].values)
    itemid = list(data['ItemId'].values)
    file = open(to_path, "w")

    sessions = []
    session = []
    sidnow = -1
    for sid, iid in zip(sessionid, itemid):
        if sidnow < 0:
            sidnow = sid
        if sid == sidnow:
            session.append(iid)
        elif sid != sidnow:
            if 10 > len(session) > 2:
                # if len(session) == 8:
                for i in range(len(session) - 1):
                    file.write(str(session[i]) + ',')
                file.write(str(session[len(session) - 1]) + '\n')
                # file.write(','*(9-len(session))+'\n')
                sessions.append(session)
            session = []
            sidnow = sid
            session.append(iid)
    file.close()
    print("write finish to " + to_path)
    return sessions


def tosessions_music(from_path, to_path):
    data = pd.read_csv(from_path, sep='\t', header=None, names=['user', 'time', 'singerid', 'singer', 'songid', 'song'])
    '''
    user_000001	2009-05-04T13:19:22Z	a7f7df4a-77d8-4f12-8acd-5c60c93f4de8	坂本龍一		Parolibre (Live_2009_4_15)
    user_000001	2009-05-04T13:13:38Z	a7f7df4a-77d8-4f12-8acd-5c60c93f4de8	坂本龍一		Bibo No Aozora (Live_2009_4_15)
    user_000001	2009-05-04T13:06:09Z	a7f7df4a-77d8-4f12-8acd-5c60c93f4de8	坂本龍一	f7c1f8f8-b935-45ed-8fc8-7def69d92a10	The Last Emperor (Theme)
    user_000001	2009-05-04T13:00:48Z	a7f7df4a-77d8-4f12-8acd-5c60c93f4de8	坂本龍一		Happyend (Live_2009_4_15)
    '''
    del data['singerid']
    del data['singer']
    del data['song']
    # data.sort_values(['user', 'time'], inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    print("data.shape", data.shape)

    userid = list(data['user'].values)
    itemid = list(data['songid'].values)
    times = list(data['time'].values)


    file = open(to_path, "w")

    session = []
    # sessions = []
    sesstime = []
    uidnow = ""

    for sid, iid, t in zip(userid, itemid, times):
        if uidnow == "":
            uidnow = sid
        if sid == uidnow:
            session.append(iid)
            sesstime.append(t)
        elif sid != uidnow:
            dt1 = time.strptime(sesstime[0], '%Y-%m-%dT%H:%M:%SZ')
            dt2 = time.strptime(sesstime[4], '%Y-%m-%dT%H:%M:%SZ')
            if dt1.tm_year != dt2.tm_year or dt1.tm_mon != dt2.tm_mon:
                break
            if dt2.tm_hour - dt1.tm_hour >= 2:
                break
            for i in range(len(session) - 1):
                file.write(str(session[i]) + ',')
            file.write(str(session[len(session) - 1]) + '\n')
            # sessions.append(session)
            session = []
            uidnow = sid
            session.append(iid)
            sesstime.append(t)
            for i in range(len(session) - 1):
                file.write(str(session[i]) + ',')
            file.write(str(session[len(session) - 1]) + '\n')
        # sessions.append(session)
    file.close()
    print("write finish to " + to_path)
    # return sessions


def music_session2small(from_path, to_path, length):
    file = open(from_path, "r")
    lines = file.readlines()
    file.close()
    file = open(to_path, "w")
    smalls = []
    for line in lines:
        iids = line.split(",")
        print("%d item in this line" % len(iids))
        index = 0
        while index < len(iids) - 5:
            for i in range(index + length - 1, index, -1):
                file.write(str(iids[i]) + ',')
            file.write(str(iids[index]) + '\n')
            smalls.append(iids[index:index + length])
            index += length
    file.close()


if __name__ == '__main__':
    # tosessions_yoo('yoo.csv', 'yoo_sessions.txt')
    tosessions_music('short-user-filter.csv', 'music_sessions.txt')
    music_session2small('music_sessions.txt', 'music_sessions_small.txt', 5)
