import matplotlib.pyplot as plt
COLORS = ['green','red','blue']
LABELS = ['Small Img','Medium Img','Large Img']
def plot_data_xy(d_list, fig_name, kind="line"):
    count=0
    for d in d_list:
        print(d)
        x, y = d["x"], d["y"]
        if(kind=="line"): plt.plot(x, y, color=COLORS[count], marker='o',
        linewidth=1, markersize=5, label=LABELS[count])
        else: plt.bar(x, y, color=COLORS[count], label=LABELS[count])
        count+=1

    
    plt.xlabel(d["x_title"])
    plt.ylabel(d["y_title"])
    plt.xticks(x)
    plt.title(fig_name)
    plt.legend(loc="upper right")
    #plt.show()
    plt.savefig(fig_name+".jpg", format="jpg")
    # linestyle='dashed',

def run_plots():
    #xy1 = {1:1983, 4:1626, 8:961, 16:758}
    #y = xy1.values()
    #x = xy1.keys()
    
    dod = []
    d = dict()
    y = [3.996799,2.533878, 2.085479,2.175043 ,2.045479,2.460318,2.326194 ,2.872909]
    x = [32, 64, 128, 156, 192, 256, 384, 512]
    d["x"], d["y"], d["x_title"], d["y_title"] =\
        x,y, "Number of Chunks", "Execution Time (sec)"
    dod.append(d)
    d = dict()
    y = [12.202852, 7.945029, 6.302318, 6.717636, 6.178567, 6.216856, 6.520382, 8.702709]
    d["x"], d["y"], d["x_title"], d["y_title"] =\
        x,y, "Number of Chunks", "Execution Time (sec)"
    dod.append(d)
    d = dict()
    y = [52.194200, 32.854623, 26.481347, 26.642569, 25.991869, 25.625183, 28.241303, 34.618534]
    d["x"], d["y"], d["x_title"], d["y_title"] =\
        x,y, "Number of Chunks", "Execution Time (sec)"
    dod.append(d)

    plot_data_xy(dod, "k-means CUDA (Chunked): Execution Times")


run_plots()

def su(L):
    LL = []
    for i in range(1, len(L)):
        LL.append(L[0]/L[i])
    print(LL)

def run():
    d = {1: 1.2, 2: 2.54, 4: 1.2, 8: 1.17, 16: 1.3, 24: 1.8, 28: 1.6}
    LL1 =[6.45, 9.7, 6.8, 5.01, 6.1, 5.9, 8.4]
    su(LL1)

#run()
