import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding,Dense,Dropout,Bidirectional,LSTM,GRU,Conv1D,Layer
# from tensorflow.keras import load_weights
from sklearn.metrics import roc_curve,auc
from keras.models import load_model


def get_feature(seq_matrix):
    """将字符编码为整数
    """
    seq_matrix = list(seq_matrix)
    # print(seq_matrix)
    ind_to_char = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
    char_to_ind = {char: i for i, char in enumerate(ind_to_char)}
    #return [ind_to_char.index(i) for i in list(seq_matrix)]
    return [char_to_ind[i] for i in seq_matrix]

#定义SN、SP、ACC、MCC
def sn_sp_acc_mcc(true_label,predict_label,pos_label=1):
    import math
    pos_num = np.sum(true_label==pos_label)
    print('pos_num=',pos_num)
    neg_num = true_label.shape[0]-pos_num
    print('neg_num=',neg_num)
    tp =np.sum((true_label==pos_label) & (predict_label==pos_label))
    print('tp=',tp)
    tn = np.sum(true_label==predict_label)-tp
    print('tn=',tn)
    sn = tp/pos_num
    sp = tn/neg_num
    acc = (tp+tn)/(pos_num+neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    print('fn=',fn)
    print('fp=',fp)
    
    tp = np.array(tp,dtype=np.float64)
    tn = np.array(tn,dtype=np.float64)
    fp = np.array(fp,dtype=np.float64)
    fn = np.array(fn,dtype=np.float64)
    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn)))
    return sn,sp,acc,mcc

class Attention3d(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention3d, self).__init__(**kwargs)

    def get_config(self):
        config = {"W_regularizer": self.W_regularizer,
                  "b_regularizer": self.b_regularizer, "W_constraint": self.W_constraint,
                  "b_constraint": self.b_constraint,
                  "bias": self.bias, "step_dim": self.step_dim, "features_dim": self.features_dim}
        base_config = super(Attention3d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=initializers.get('glorot_uniform'),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
    
def define_model():
    maxlen = 33
    max_features = 21
    embedding_dims = 64
    class_num = 1
    last_activation = 'sigmoid'
    input = Input((maxlen,))
    embedding = Embedding(max_features, embedding_dims, input_length=maxlen)(input)

    x = Bidirectional(GRU(64, return_sequences=True))(embedding)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
#     x = Bidirectional(GRU(16, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Attention3d(maxlen)(x)

    t = Dense(16,activation='relu')(x)
    output = Dense(class_num, activation=last_activation)(t)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model
with  open("data\\tytest21.fa") as f:
    ST_test = f.readlines()
    ST_test = [s.strip() for s in ST_test]

def remove_name(data):
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new

def run_predict(id_seqs):
    id = []
    seqs = []
    site = []
    is_phosphory = []
    prob = []
    seq_len = len(id_seqs)
    for i in range(seq_len):
        record = id_seqs[i]
        if len(record) >= 33:
            for j in range(len(record)-32):
                seq = record[j:j+33]
                seq = seq.upper()
                for seq_one in seq:
                    if seq_one in 'ACDEFGHIKLMNPQRSTVWYX':
                        continue
                    else:
                        return "Please enter the sequence of 20 amino acids or 'X'"
                if seq[16] in "ST":
                    fea_df = get_feature(seq)
                    feature = np.array(fea_df)
                    feature = np.expand_dims(feature, axis=0)
                    model = define_model()
                    model.load_weights('model/ST_model.h5')
                    res = model.predict(feature)
                    id.append(id_seqs[i-1].split('>')[-1])
                    seqs.append(seq)
                    # print(1, seq[17])
                    site.append(seq[16])
                    prob.append(res)
                    if res > 0.5:
                        is_phosphory.append("True")
                    else:
                        is_phosphory.append("False")
                elif seq[16] == "Y":
                    # print(2, seq[17])
                    fea_df = get_feature(seq)
                    feature = np.array(fea_df)
                    feature = np.expand_dims(feature, axis=0)
                    model = define_model()
                    model.load_weights('model/Y_model.h5')
                    res = model.predict(feature)
                    id.append(id_seqs[i - 1].split('>')[-1])
                    seqs.append(seq)
                    site.append("Y")
                    prob.append(res)
                    if res > 0.5:
                        is_phosphory.append("True")
                    else:
                        is_phosphory.append("False")
                else:
                    continue
        else:
            id.append(id_seqs[i - 1].split('>')[-1])
            seqs.append("Sequence length must be >= 33")
            site.append("-")
            is_phosphory.append("-")
            prob.append("-")
    res_df = pd.DataFrame(columns=['id', 'seq', 'site', 'is_phosphory', 'prob'])
    res_df.id = id
    res_df.seq = seqs
    res_df.site = site
    res_df.is_phosphory = is_phosphory
    res_df.prob = prob
    return res_df,prob

if __name__ == '__main__':
    with  open("data\ST-test.fa") as f:
        Y_train = f.readlines()
        Y_train = [s.strip() for s in Y_train]
    ST_test_x = remove_name(ST_test)
    # print(len(ST_train_x),len(ST_train_x[0]))
    print(len(ST_test_x),len(ST_test_x[0]))

    # ST_train_y = np.concatenate([np.ones((int(len(ST_train_x)/2),)),np.zeros((int(len(ST_train_x)/2),))], axis=0)  #竖向拼接
    ST_test_y = np.concatenate([np.ones((int(len(ST_test_x)/2),)),np.zeros((int(len(ST_test_x)/2),))], axis=0)
    print(ST_test_y.shape)

    # data = "ST-train.fa"
    predict_result,prob = run_predict(Y_train)
    prob=np.array(prob)
    pred1 = np.squeeze(prob,axis=-1)
    f1 = pred1>0.5
    pred1[f1]=1
    pred1[pred1<0.6]=0
    sn_sp_acc_mcc_5fold1 = sn_sp_acc_mcc(ST_test_y,pred1,pos_label=1)
    print(sn_sp_acc_mcc_5fold1)
    FPR1,TPR1,threshold = roc_curve(ST_test_y,prob,pos_label=1)
    AUC1 = auc(FPR1,TPR1)
    print(AUC1)

    print(predict_result)