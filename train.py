import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from tensorflow import keras
from keras.optimizers import adam_v2
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from keras.regularizers import l1, l2,l1_l2
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding,Dense,Dropout,Bidirectional,LSTM,GRU,Conv1D,Layer,BatchNormalization,Flatten,MaxPooling1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,auc,confusion_matrix,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
with  open("data\\coiltrain.fa") as f:
    ST_train = f.readlines()
    ST_train = [s.strip() for s in ST_train]
with  open("data\\coiltest.fa") as f:
    ST_test = f.readlines()
    ST_test = [s.strip() for s in ST_test]
print(len(ST_train),len(ST_test))

def remove_name(data):
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new

ST_train_x = remove_name(ST_train)
ST_test_x = remove_name(ST_test)
print(len(ST_train_x),len(ST_train_x[0]))
print(len(ST_test_x),len(ST_test_x[0]))

ST_train_y = np.concatenate([np.ones((int(len(ST_train_x)/2),)),np.zeros((int(len(ST_train_x)/2),))], axis=0)  #竖向拼接
ST_test_y = np.concatenate([np.ones((int(len(ST_test_x)/2),)),np.zeros((int(len(ST_test_x)/2),))], axis=0)
print(ST_train_y.shape,ST_test_y.shape)

def encode_matrix(seq_matrix):
    ind_to_char = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','O']
    char_to_ind = {char: i for i, char in enumerate(ind_to_char)}
    return [[char_to_ind[i] for i in s] for s in seq_matrix]

ST_train_x = encode_matrix(ST_train_x)
ST_test_x = encode_matrix(ST_test_x)
ST_train_x = np.array(ST_train_x)
ST_test_x = np.array(ST_test_x)
print(ST_train_x.shape,ST_test_x.shape)

def visualize_tsne(data, labels):
    tsne = TSNE(n_components=2, perplexity=10,learning_rate=100)
    tsne_results = tsne.fit_transform(data)
    plt.scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1],s=20, c='r', alpha=0.8,label='pos')
    plt.scatter(tsne_results[labels==0, 0], tsne_results[labels==0, 1],s=20, c='b', alpha=0.8,label='neg')
    plt.legend()

def metric(true_label,res,pos_label=1):
    import math
    predict_label = np.squeeze(res,axis=-1)
    f1 = predict_label>0.5
    predict_label[f1]=1
    predict_label[predict_label<0.6]=0
    pos = np.sum(true_label==pos_label)
    print('pos=',pos)
    neg = true_label.shape[0]-pos
    print('neg=',neg)
    tp =np.sum((true_label==pos_label) & (predict_label==pos_label))
    print('tp=',tp)
    tn = np.sum(true_label==predict_label)-tp
    print('tn=',tn)
    sn = tp/pos
    sp = tn/neg
    acc = (tp+tn)/(pos+neg)
    fn = pos - tp
    fp = neg - tn
    print('fn=',fn)
    print('fp=',fp)
    
    tp = np.array(tp,dtype=np.float64)
    tn = np.array(tn,dtype=np.float64)
    fp = np.array(fp,dtype=np.float64)
    fn = np.array(fn,dtype=np.float64)
    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn)))
    pre = tp / (tp + fp)
    FPR1,TPR1,threshold = roc_curve(true_label,res,pos_label=1)
    AUC1 = auc(FPR1,TPR1)
    return sn,sp,pre,acc,mcc,AUC1

class SelfAttention(Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.Wq = Dense(units,activation='elu')
        self.Wk = Dense(units,activation='elu')
        self.Wv = Dense(units,activation='elu')
        self.dense = Dense(units,activation='elu')
    def call(self, inputs):
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        attention_output = self.dense(attention_output)
        return attention_output

def save_metrics(filename, metric,auc,aupr,i):
    header = ['Fold', 'SN', 'SP', 'PRE', 'ACC', 'MCC', 'AUROC', 'AUPRC']
    with open(filename, 'a+', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # 检查文件是否为空，如果是则写入表头
            writer.writerow(header)
        fold = i + 1
        sn =  "{:.4f}".format(metric[0])
        sp =  "{:.4f}".format(metric[1])
        pre = "{:.4f}".format(metric[2])
        acc = "{:.4f}".format(metric[3])
        mcc = "{:.4f}".format(metric[4])
        auroc = "{:.4f}".format(auc)
        auprc = "{:.4f}".format(aupr)
        row = [fold, sn, sp, pre, acc, mcc, auroc, auprc]
        writer.writerow(row)

def define_model():
    maxlen = 21
    max_features = 21
    embedding_dims = 64
    class_num = 1
    input = Input((maxlen,))
    x = Embedding(max_features, embedding_dims, input_length=maxlen)(input)
    x = Dropout(0.5)(x)
    x=Conv1D(filters=64,kernel_size=2,activation='elu')(x)
    x=MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = Bidirectional(GRU(16, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    # x = SelfAttention(units=maxlen)(x)
    x=Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(16,activation='elu',kernel_initializer='he_normal',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x)
    output = Dense(class_num, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    optimizer = adam_v2.Adam(learning_rate=1e-4, epsilon=1e-8)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model


#交叉验证
def Kfold(data_x,data_y,testx,testy,K):
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=6)

    for i,(train, test) in enumerate(kfold.split(data_x, data_y)):
        print('\nfold%d'%(i+1))
        checkpoint = [
            keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='auto', restore_best_weights = True),
            keras.callbacks.CSVLogger('train_loss/coil/tlo_%d.log' %i, separator=',', append=False),
        ]
        model = define_model()
        model.summary()
        history=model.fit(data_x[train],data_y[train],batch_size=32,epochs=200,validation_data=(data_x[test],data_y[test]),shuffle=True,callbacks=[checkpoint],verbose=1)
        model.save_weights('./model/coil/weight%d.h5'%i)
        
        loss=history.history['loss']
        val_loss=history.history['val_loss']
        accuracy=history.history['accuracy']
        val_accuracy=history.history['val_accuracy']
        plt.plot(loss,label='Training Loss')
        plt.plot(val_loss,label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('./loss_img/coil/%d'%i)
        # plt.show()
        plt.close()
        plt.plot(accuracy,label='Training accuracy')
        plt.plot(val_accuracy,label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig('./acc_img/coil/%d'%i)
        # plt.show()  
        plt.close()
        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[9].output)
        intermediate_output = intermediate_layer_model.predict(data_x[train])
        df0 = pd.DataFrame(intermediate_output)
        df0.to_csv('tsne/prob/coil/%d.csv'%i, index=False, header=False)
        visualize_tsne(data_x[train], data_y[train])
        plt.title('The input layer')
        plt.savefig('./tsne/img/coil/before_%d'%i)
        plt.close()
        # plt.show()
        visualize_tsne(intermediate_output, data_y[train])
        plt.title('The flatten dense layer')
        plt.savefig('./tsne/img/coil/after_%d'%i)
        plt.close()
        # plt.show() 

        res = model.predict(data_x[test])
        df = pd.DataFrame( model.predict(data_x[test]))
        df.to_csv('pre_pro/train/coil/%d.csv'%i, index=False, header=False)
        valid_metric = metric(data_y[test],res,pos_label=1)
        pre, rec, threshold = precision_recall_curve(data_y[test],model.predict(data_x[test]),pos_label=1)
        AUPRC = auc(rec, pre)
        FPR,TPR,threshold = roc_curve(data_y[test],model.predict(data_x[test]),pos_label=1)
        AUC = auc(FPR,TPR)
        print(valid_metric,AUC,AUPRC)
        save_metrics('metrics/train/coil.csv', valid_metric,AUC,AUPRC,i)

        res1=model.predict(testx)
        df1 = pd.DataFrame(model.predict(testx))
        df1.to_csv('pre_pro/test/coil/%d.csv'%i, index=False, header=False)
        test_metric = metric(testy,res1,pos_label=1)
        pre1, rec1, threshold = precision_recall_curve(testy,model.predict(testx),pos_label=1)
        AUPRC1 = auc(rec1, pre1)
        FPR1,TPR1,threshold = roc_curve(testy,model.predict(testx),pos_label=1)
        AUC1 = auc(FPR1,TPR1)
        print(test_metric,AUC1,AUPRC1)
        save_metrics('metrics/test/coil.csv', test_metric,AUC1,AUPRC1,i)
        #   ROC和pre曲线
        # plt.plot(rec1, pre1, marker='.', label='Precision-Recall Curve')
        # plt.title('Precision-Recall Curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.show()
        # plt.plot(FPR1, TPR1, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % AUC1)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()
        
Kfold(ST_train_x,ST_train_y,ST_test_x,ST_test_y,10)
