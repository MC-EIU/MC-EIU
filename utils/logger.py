import time
import os
import logging
import fcntl

def get_logger(path, suffix):
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S',time.localtime(time.time()))
    logger = logging.getLogger(__name__+cur_time)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class ResultRecorder(object):
    def __init__(self, path, total_cv=10):
        self.path = path
        self.total_cv = total_cv
        if not os.path.exists(self.path):
            f = open(self.path, 'w')
            # f.write('acc\tuar\tf1\n')
            f.write('emo_acc\tint_acc\temo_uar\tint_uar\n')
            f.close()
    
    def is_full(self, content):
        if len(content) < self.total_cv+1:
            return False
        
        for line in content:
            if not len(line.split('\t')) == 4:
                return False
        return True
    
    def calc_mean(self, content):
        # acc = [float(line.split('\t')[0]) for line in content[1:]]
        # uar = [float(line.split('\t')[1]) for line in content[1:]]
        # f1 = [float(line.split('\t')[2]) for line in content[1:]]
        # mean_acc = sum(acc) / len(acc)
        # mean_uar = sum(uar) / len(uar)
        # mean_f1 = sum(f1) / len(f1)
        # return mean_acc, mean_uar, mean_f1
        emo_acc = [float(line.split('\t')[0]) for line in content[1:]]
        int_acc = [float(line.split('\t')[1]) for line in content[1:]]
        emo_uar = [float(line.split('\t')[2]) for line in content[1:]]
        int_uar = [float(line.split('\t')[3]) for line in content[1:]]
        mean_emo_acc = sum(emo_acc) / len(emo_acc)
        mean_int_acc = sum(int_acc) / len(int_acc)
        mean_emo_uar = sum(emo_uar) / len(emo_uar)
        mean_int_uar = sum(int_uar) / len(int_uar)
        return mean_emo_acc, mean_int_acc, mean_emo_uar, mean_int_uar

    def write_result_to_tsv(self, results, cvNo):
        # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
        f_in = open(self.path)
        fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
        content = f_in.readlines()
        if len(content) < self.total_cv+1:
            content += ['\n'] * (self.total_cv-len(content)+1)
        keys = [item for item in results.keys()]
        content[cvNo] = '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results[keys[0]], results[keys[1]], results[keys[2]], results[keys[3]])
        if self.is_full(content):
            mean_emo_acc, mean_int_acc, mean_emo_uar, mean_int_uar = self.calc_mean(content)
            content.append('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_emo_acc, mean_int_acc, mean_emo_uar, mean_int_uar))

        f_out = open(self.path, 'w')
        f_out.writelines(content)
        f_out.close()
        f_in.close()                              # 释放锁


class LossRecorder(object):
    def __init__(self, path, total_cv=10, total_epoch=40):
        self.path = path
        self.total_epoch = total_epoch
        self.total_cv = total_cv
        if not os.path.exists(self.path):
            f = open(self.path, 'w')
            f.close()

    def is_full(self, content):
        if len(content) < self.total_cv + 1:
            return False

        for line in content:
            if not len(line.split('\t')) == 3:
                return False
        return True

    def calc_mean(self, content):
        loss_list = [[] * self.total_cv] * self.total_epoch
        mean_list = [[] * self.total_cv] * self.total_epoch
        for i in range(0, self.total_epoch):
            loss_list[i] = [float(line.split('\t')[i]) for line in content[1:]]
        for i in range(0, self.total_epoch):
            mean_list[i] = sum(loss_list[i]) / len(loss_list[i])
        return mean_list

    def write_result_to_tsv(self, results, cvNo):
        # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
        f_in = open(self.path)
        fcntl.flock(f_in.fileno(), fcntl.LOCK_EX)  # 加锁
        content = f_in.readlines()
        if len(content) < self.total_cv + 1:
            content += ['\n'] * (self.total_cv - len(content) + 1)
        string = ''
        for i in results:
            string += str(i.numpy())[:8]
            string += '\t'
        content[cvNo] = string + '\n'

        f_out = open(self.path, 'w')
        f_out.writelines(content)
        f_out.close()
        f_in.close()  # 释放锁

    def read_result_from_tsv(self,):
        f_out = open(self.path)
        fcntl.flock(f_out.fileno(), fcntl.LOCK_EX)
        content = f_out.readlines()
        loss_list = [[] * self.total_cv] * self.total_epoch
        for i in range(0, self.total_epoch):
            loss_list[i] = [float(line.split('\t')[i]) for line in content[1:]]
        mean = self.calc_mean(content)
        return mean
