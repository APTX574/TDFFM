import datetime


def log(net_name, epoch, index_type, att_type, loss, acc, bz, lr, step):
    path = "./logs"
    coding = index_type[0]
    for i in range(len(index_type)-1):
        coding = coding + "+" + index_type[i+1]
    with open(path, "a") as file:
        time_ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        str_ = "{},{},{},{},{},{},{},{},{},{}\n".format(time_, net_name, coding, att_type, epoch, loss, acc, bz, lr,
                                                        step)
        file.write(str_)
