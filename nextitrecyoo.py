import tensorflow as tf
import data_loader_recsys
import generator_recsys
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse
from Data.Session.load_yoo import yoosortlen


# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
# if you are dealing very huge industry dataset, e.g.,several hundred million items, you may have memory problem during training, but it 
# be easily solved by simply changing the last layer, you do not need to calculate the cross entropy loss
# based on the whole item vector. Similarly, you can also change the last layer (use tf.nn.embedding_lookup or gather) in the prediction phrase 
# if you want to just rank the recalled items instead of all items. The current code should be okay if the item size < 5 million.



# Strongly suggest running codes on GPU with more than 10G memory!!!
# if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print(x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print("generating subsessions is done!")
    return x_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    # history_sequences_20181014_fajie_smalltest.csv
    parser.add_argument('--datapath', type=str, default='Data/Session/yoo.csv',
                        # parser.add_argument('--datapath', type=str, default='Data/Session/user-filter-20000items-session5.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=100,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=10000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    args = parser.parse_args()

    if args.datapath == 'Data/Session/yoo.csv':
        train_set, valid_set, item2idx = yoosortlen(args.datapath)
        items = item2idx.keys()
    else:
        dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
        all_samples = dl.item
        items = dl.item_dict

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
        all_samples = all_samples[shuffle_indices]

        # Split train/test set
        dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
        train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_generatesubsession:
        train_set = generatesubsequence(train_set)

    model_para = {
        # if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items),
        'dilated_channels': 100,
        # if you use nextitnet_residual_block, you can use [1, 4, ],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        'dilations': [1, 2, 4],
        'kernel_size': 3,
        'learning_rate': 0.005,
        'batch_size': 32,
        'iterations': 256,
        'is_negsample': False  # False denotes no negative sampling
    }

    print("len(items), also sample number of songs:", len(items))
    print("is_generatesubsession:", args.is_generatesubsession)
    print("datapath:", args.datapath)
    print("dilations:", model_para['dilations'])
    print("learning_rate:", model_para['learning_rate'])

    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph(model_para['is_negsample'], reuse=True)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for iter in range(model_para['iterations']):

        # train
        train_loss = []
        batch_size = model_para['batch_size']
        start = time.time()
        for lenkey in train_set:
            batch_no_train = 0
            idss = train_set[lenkey]
            print(lenkey, "total train batch:", round(len(idss) / batch_size))
            while (batch_no_train + 1) * batch_size < len(idss):
                item_batch = idss[batch_no_train * batch_size: (batch_no_train + 1) * batch_size]
                _, loss, results = sess.run(
                    [optimizer, itemrec.loss,
                     itemrec.arg_max_prediction],
                    feed_dict={
                        itemrec.itemseq_input: item_batch
                    })

                train_loss.append(loss)
                batch_no_train += 1
        end = time.time()
        print("train LOSS: %.4f, time: %.2fs" % (np.mean(train_loss), end - start))

        # test
        test_loss = []
        formrr5, forhit5, forndcg5, formrr20, forhit20, forndcg20 = [], [], [], [], [], []
        maxmrr5, maxmrr20, maxhit5, maxhit20, maxndcg5, maxndcg20 = 0, 0, 0, 0, 0, 0
        start = time.time()
        for lenkey in train_set:
            batch_no_test = 0
            idss = valid_set[lenkey]
            print(lenkey, "total test batch:", round(len(idss) / batch_size))
            while (batch_no_test + 1) * batch_size < len(idss):
                item_batch = idss[batch_no_test * batch_size: (batch_no_test + 1) * batch_size]
                [probs], loss = sess.run(
                    [[itemrec.g_probs], [itemrec.loss_test]],
                    feed_dict={
                        itemrec.input_predict: item_batch
                    })
                test_loss.append(loss)
                batch_no_test += 1

                for bi in range(probs.shape[0]):
                    pred_items_5 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k)  # top_k=5
                    pred_items_20 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k + 15)

                    true_item = item_batch[bi][-1]
                    predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
                    pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

                    rank_5 = predictmap_5.get(true_item)
                    rank_20 = pred_items_20.get(true_item)
                    if rank_5 == None:
                        formrr5.append(0.0)
                        forhit5.append(0.0)  # 2
                        forndcg5.append(0.0)  # 2
                    else:
                        MRR_5 = 1.0 / (rank_5 + 1)
                        Rec_5 = 1.0  # 3
                        ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                        formrr5.append(MRR_5)
                        forhit5.append(Rec_5)  # 4
                        forndcg5.append(ndcg_5)  # 4
                    if rank_20 == None:
                        formrr20.append(0.0)
                        forhit20.append(0.0)  # 2
                        forndcg20.append(0.0)  # 2
                    else:
                        MRR_20 = 1.0 / (rank_20 + 1)
                        Rec_20 = 1.0  # 3
                        ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
                        formrr20.append(MRR_20)
                        forhit20.append(Rec_20)  # 4
                        forndcg20.append(ndcg_20)  # 4

                thismrr5 = sum(formrr5) / float(len(formrr5))
                thismrr20 = sum(formrr20) / float(len(formrr20))
                thishit5 = sum(forhit5) / float(len(forhit5))
                thishit20 = sum(forhit20) / float(len(forhit20))
                thisndcg5 = sum(forndcg5) / float(len(forndcg5))
                thisndcg20 = sum(forndcg20) / float(len(forndcg20))

                if thisndcg5 > maxndcg5:
                    maxndcg5 = thisndcg5
                    maxmrr5 = thismrr5
                    maxhit5 = thishit5
                if thisndcg20 > maxndcg20:
                    maxndcg20 = thisndcg20
                    maxmrr20 = thismrr20
                    maxhit20 = thishit20

                batch_no_test += 1
                # print("BATCH_NO: {}".format(batch_no_test))
        end = time.time()
        print("train LOSS: %.4f, time: %.2fs" % (np.mean(test_loss), end - start))

        print("\t\tmax_mrr_5=%.4f   max_hit_5=%.4f   max_ndcg_5=%.4f" %
              (maxmrr5, maxhit5, maxndcg5))  # 5
        print("\t\tmax_mrr_20=%.4f  max_hit_20=%.4f  max_ndcg_20=%.4f\n" %
              (maxmrr20, maxhit20, maxndcg20))  # 20


if __name__ == '__main__':
    main()
